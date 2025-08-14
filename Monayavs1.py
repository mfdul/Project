from collections import defaultdict
from simulator import Simulator
from planners import Planner
from problems import HealthcareProblem, ResourceType
from reporter import EventLogReporter, ResourceScheduleReporter

# ---------------- Tunable knobs ----------------
ADMIT_HORIZON_HOURS = 48        # plan >= 24h ahead (buffer = 48h)
SAFE_REPLAN_DAYS = 14           # don't replan < 14 days before original time
LOOKAHEAD_HOURS = 7 * 24        # search window when looking for admission slots

# Preferred intake window on weekdays
PREFERRED_START_HOUR = 8        # 08:00
PREFERRED_END_HOUR = 12         # until 12:00

# Replanning benefit thresholds
IMPROVEMENT_EARLIER_HOURS = 24  # move if at least this much earlier
LOAD_IMPROVEMENT_FACTOR = 0.5   # or if new_slot load ratio <= 50% of old

WEEK = 168

# Resource maxima (from project statement)
MAX_OR = 5
MAX_A_BED = 30
MAX_B_BED = 40
MAX_INTAKE = 4
MAX_ER = 9


class HeuristicPlanner(Planner):
    """
    Capacity-aware hospital planner:
      - Schedules a weekday day/night template one week ahead (t+158 rule),
        respecting resource maxima and "only increase < 1 week" constraint.
      - Maintains an admission quota per hour (derived from INTAKE day/night levels).
      - Plans new cases with Longest-Waiting-First into the earliest hour that still has slack,
        preferring weekday mornings and lower load ratio.
      - Replans only when safe (> 14 days before original time) and beneficial.
    """

    def __init__(self, eventlog_file="./temp/event_log.csv", data_columns=("diagnosis",)):
        super().__init__()
        self.eventlog_reporter = EventLogReporter(eventlog_file, data_columns)
        self.resource_reporter = ResourceScheduleReporter()

        # Admission calendar (how many we allow vs. already booked per hour)
        self.admit_quota  = defaultdict(int)  # hour -> allowed number of admissions
        self.admit_booked = defaultdict(int)  # hour -> already booked

        # Track plans and first-seen times (for priority by waiting)
        self.planned_time = {}                # case_id -> planned hour
        self.arrival_time = {}                # case_id -> first seen timestamp

        self.bootstrapped_until = -1          # last hour (exclusive) through which we built schedule/quota

    # ---------------- REPORT ----------------
    def report(self, case_id, element, timestamp, resource, lifecycle_state, data=None):
        self.eventlog_reporter.callback(case_id, element, timestamp, resource, lifecycle_state)
        self.resource_reporter.callback(case_id, element, timestamp, resource, lifecycle_state, data)
        if element == "arrival" and lifecycle_state == "complete":
            self.arrival_time.setdefault(case_id, int(timestamp))

    # --------------- SCHEDULE ---------------
    def schedule(self, simulation_time: int):
        """
        Daily at 18:00. Emit resource changes starting from t+158 (08:00 next same weekday)
        for the coming week. Weekdays: day (08–18) high levels, nights/weekends low.
        Also (re)prime admission quotas for the next week (>= t+14), only increasing them.
        """
        t = int(simulation_time)
        changes = []

        # Build day/night plan for next week from 08:00 one week ahead
        start_next_week_8 = t + 158
        horizon_end = start_next_week_8 + WEEK
        if horizon_end > self.bootstrapped_until:
            for d in range(7):
                day_start = start_next_week_8 + d * 24   # 08:00
                night_start = day_start + 10              # 18:00
                is_weekday = ((day_start % WEEK) // 24) < 5

                if is_weekday:
                    changes.extend([
                        (ResourceType.OR, day_start, min(5, MAX_OR)),
                        (ResourceType.A_BED, day_start, min(30, MAX_A_BED)),
                        (ResourceType.B_BED, day_start, min(40, MAX_B_BED)),
                        (ResourceType.INTAKE, day_start, min(4, MAX_INTAKE)),
                        (ResourceType.ER_PRACTITIONER, day_start, min(9, MAX_ER)),
                        (ResourceType.OR, night_start, 1),
                        (ResourceType.INTAKE, night_start, 1),
                    ])
                else:
                    # weekend: keep low all day (single change at 08:00)
                    changes.extend([
                        (ResourceType.OR, day_start, 1),
                        (ResourceType.A_BED, day_start, min(30, MAX_A_BED)),
                        (ResourceType.B_BED, day_start, min(40, MAX_B_BED)),
                        (ResourceType.INTAKE, day_start, 1),
                        (ResourceType.ER_PRACTITIONER, day_start, min(9, MAX_ER)),
                    ])

            self.bootstrapped_until = horizon_end

        # Admission quota for next week (>= t+14); only increases allowed
        for offset in range(14, WEEK + 14):
            h = t + offset
            hod = h % 24
            weekday = ((h % WEEK) // 24) < 5
            base_quota = 4 if (weekday and 8 <= hod < 16) else 1
            self.admit_quota[h] = max(self.admit_quota.get(h, 0), base_quota)

        return changes

    # ----------------- PLAN -----------------
    def plan(self, cases_to_plan, cases_to_replan, simulation_time: int):
        """
        Heuristic plan:
          - Longest-waiting-first priority
          - Capacity-aware: book only where quota > booked
          - Prefer weekday mornings (08:00–12:00)
          - Look up to LOOKAHEAD_HOURS
          - Replan only when safe (>14d) and beneficial (earlier or much less loaded)
        """
        decisions = []
        now = int(simulation_time)

        def waiting_key(cid):
            arr = self.arrival_time.get(cid, now)
            return now - arr

        # New cases
        for cid in sorted(cases_to_plan, key=waiting_key, reverse=True):
            earliest = now + ADMIT_HORIZON_HOURS
            slot = self._choose_slot(earliest, prefer_mornings=True)
            if slot is not None:
                decisions.append((cid, slot))
                self.planned_time[cid] = slot
                self.admit_booked[slot] += 1

        # Replanning (safe & beneficial)
        for cid in sorted(cases_to_replan, key=waiting_key, reverse=True):
            prev = self.planned_time.get(cid)
            if prev is None:
                earliest = now + ADMIT_HORIZON_HOURS
                slot = self._choose_slot(earliest, prefer_mornings=True)
                if slot is not None:
                    decisions.append((cid, slot))
                    self.planned_time[cid] = slot
                    self.admit_booked[slot] += 1
                continue

            if prev - now < SAFE_REPLAN_DAYS * 24:
                continue

            prev_ratio = self._slot_load_ratio(prev)
            earliest = now + ADMIT_HORIZON_HOURS
            best = self._choose_slot(earliest, prefer_mornings=True)
            if best is None:
                continue

            earlier_enough = (prev - best) >= IMPROVEMENT_EARLIER_HOURS
            much_less_loaded = (self._slot_load_ratio(best) <= LOAD_IMPROVEMENT_FACTOR * prev_ratio)

            if earlier_enough or much_less_loaded:
                decisions.append((cid, best))
                self.admit_booked[best] += 1
                self.planned_time[cid] = best

        return decisions

    # ------------- Helpers ------------------
    def _choose_slot(self, earliest_hour: int, prefer_mornings: bool = True):
        """
        Find the best admission hour >= earliest_hour within LOOKAHEAD_HOURS.
        Preference:
          1) earliest hour with slack>0 inside preferred weekday morning window (08–12)
          2) else earliest hour with slack>0 anywhere
          3) tie-breaker: lower load ratio (booked/quota)
        """
        start = int(earliest_hour)
        end = start + LOOKAHEAD_HOURS

        def has_slack(h):
            q = self.admit_quota.get(h, 0)
            b = self.admit_booked.get(h, 0)
            return q > b

        def is_preferred(h):
            if not prefer_mornings:
                return False
            hod = h % 24
            weekday = ((h % WEEK) // 24) < 5
            return weekday and (PREFERRED_START_HOUR <= hod < PREFERRED_END_HOUR)

        best_h, best_ratio = None, 1e9

        # Preferred windows first
        for h in range(start, end):
            if has_slack(h) and is_preferred(h):
                ratio = self._slot_load_ratio(h)
                if ratio < best_ratio:
                    best_h, best_ratio = h, ratio
        if best_h is not None:
            return best_h

        # Any window
        for h in range(start, end):
            if has_slack(h):
                ratio = self._slot_load_ratio(h)
                if ratio < best_ratio:
                    best_h, best_ratio = h, ratio
        return best_h

    def _slot_load_ratio(self, h: int):
        q = self.admit_quota.get(h, 0)
        b = self.admit_booked.get(h, 0)
        if q <= 0:
            return 1e9
        return b / q


if __name__ == '__main__':
    # Choose how many hours to simulate; keep it short while tuning
    RUN_HOURS = 14 * 24  # 14 days; later switch to 365*24 for the final run

    planner = HeuristicPlanner("./temp/event_log.csv", ["diagnosis"])
    problem = HealthcareProblem()
    simulator = Simulator(planner, problem)

    result = simulator.run(RUN_HOURS)
    print(result)

    # SAFE plot: use exactly the hours you simulated
    planner.resource_reporter.create_graph(0, RUN_HOURS)
