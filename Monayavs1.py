from collections import defaultdict
from simulator import Simulator
from planners import Planner
from problems import HealthcareProblem, ResourceType
from reporter import EventLogReporter, ResourceScheduleReporter

# ---------------- Tunable knobs ----------------
ADMIT_HORIZON_HOURS = 24        # plan >= 24h ahead (buffer)
SAFE_REPLAN_DAYS = 14           # don't replan < 14 days before original time
LOOKAHEAD_HOURS = 14 * 24        # search window for admission slots

# Preferred intake window on weekdays
PREFERRED_START_HOUR = 8        # 08:00
PREFERRED_END_HOUR = 12         # until 12:00

# Replanning benefit thresholds
IMPROVEMENT_EARLIER_HOURS = 24  # replan if at least this much earlier
LOAD_IMPROVEMENT_FACTOR = 0.5   # or if new_slot load ratio <= 50% of old

WEEK = 168

# Resource maxima (from problems.py - HealthcareProblem.__create_resources)
MAX_OR = 5
MAX_A_BED = 30
MAX_B_BED = 40
MAX_INTAKE = 4
MAX_ER = 9

# ---------------- Benchmark (from example planner you already ran) ----------------
BENCHMARK_RESULTS = {
    "waiting_time_for_admission": 286663.61610991484,
    "waiting_time_in_hospital": 4732777.025815763,
    "nervousness": 2950031.0,
    "personnel_cost": 733401
}


class HeuristicPlanner(Planner):
    """
    Capacity-aware hospital planner:
      - Weekly day/night template one week ahead (t+158 rule) within resource maxima.
      - Admission quota per hour (higher in weekday mornings) – only increases < 1 week.
      - Plan with Longest-Waiting-First into earliest low-load hour with slack.
      - Replan only when safe (>14d) and beneficial (earlier or much lower load).
    """

    def __init__(self, eventlog_file="./temp/event_log.csv", data_columns=("diagnosis",)):
        super().__init__()
        self.eventlog_reporter = EventLogReporter(eventlog_file, data_columns)
        self.resource_reporter = ResourceScheduleReporter()

        # Admission calendar
        self.admit_quota  = defaultdict(int)  # hour -> allowed admissions
        self.admit_booked = defaultdict(int)  # hour -> booked admissions

        # Tracking
        self.planned_time = {}                # case_id -> planned hour
        self.arrival_time = {}                # case_id -> first-seen timestamp

        self.bootstrapped_until = -1          # last hour (exclusive) fully scheduled

    # ---------------- REPORT ----------------
    def report(self, case_id, element, timestamp, resource, lifecycle_state, data=None):
        self.eventlog_reporter.callback(case_id, element, timestamp, resource, lifecycle_state)
        self.resource_reporter.callback(case_id, element, timestamp, resource, lifecycle_state, data)
        # Track arrival times for waiting calculations
        if element is not None and element.label == "patient_referal" and lifecycle_state.name == "COMPLETE_EVENT":
            self.arrival_time.setdefault(case_id, int(timestamp))

    # --------------- SCHEDULE ---------------
    def schedule(self, simulation_time: int):
        """
        Adaptive weekly scheduler:
          - Publishes next-week (t+158) day/night template.
          - For each weekday in next week, picks INTAKE/OR levels by expected demand
            (derived from how many admissions are already booked that day).
          - Keeps nights and weekends minimal.
          - Primes admission quotas (>= t+14) to match chosen daytime INTAKE levels (only increases).
        """
        t = int(simulation_time)
        changes = []
        WEEK = 168

        # ---------- helper: expected demand per day (already-booked admissions) ----------
        def day_bounds(start_8):
            """returns (day_08, day_18, night_18, next_day_08) absolute hours"""
            day_08 = start_8
            day_18 = start_8 + 10
            night_18 = day_18
            next_08 = start_8 + 24
            return day_08, day_18, night_18, next_08

        def booked_in_window(h0, h1):
            """count admissions already booked in [h0, h1)"""
            total = 0
            for h in range(h0, h1):
                total += self.admit_booked.get(h, 0)
            return total

        # ---------- 1) Next-week frame starting 08:00 one week ahead ----------
        start_next_week_8 = t + 158
        horizon_end = start_next_week_8 + WEEK
        if horizon_end > self.bootstrapped_until:
            for d in range(7):
                day_start_08 = start_next_week_8 + d * 24
                day_08, day_18, night_18, next_08 = day_bounds(day_start_08)

                dow = ((day_start_08 % WEEK) // 24)  # 0..6 (Mon..Sun)
                is_weekday = dow < 5

                if not is_weekday:
                    # Weekend: minimal all day
                    changes.extend([
                        (ResourceType.OR, day_08, 1),
                        (ResourceType.A_BED, day_08, min(30, MAX_A_BED)),
                        (ResourceType.B_BED, day_08, min(40, MAX_B_BED)),
                        (ResourceType.INTAKE, day_08, 1),
                        (ResourceType.ER_PRACTITIONER, day_08, min(9, MAX_ER)),
                    ])
                    continue

                # Weekday: choose levels by demand already booked for this day's daytime (08–18)
                day_booked = booked_in_window(day_08, day_18)
                if day_booked <= 12:
                    intake_day, or_day = 2, 2
                elif day_booked <= 24:
                    intake_day, or_day = 3, 4
                else:
                    intake_day, or_day = 4, 5

                # Day shift (08–18)
                changes.extend([
                    (ResourceType.OR, day_08, min(or_day, MAX_OR)),
                    (ResourceType.A_BED, day_08, min(30, MAX_A_BED)),
                    (ResourceType.B_BED, day_08, min(40, MAX_B_BED)),
                    (ResourceType.INTAKE, day_08, min(intake_day, MAX_INTAKE)),
                    (ResourceType.ER_PRACTITIONER, day_08, min(9, MAX_ER)),
                ])
                # Night shift (18–08 next day): minimal OR/INTAKE
                changes.extend([
                    (ResourceType.OR, night_18, 1),
                    (ResourceType.INTAKE, night_18, 1),
                ])

            self.bootstrapped_until = horizon_end

        # ---------- 2) Prime admission quota (>= t+14) to match chosen daytime levels ----------
        for offset_day in range(0, 7):
            this_day_08 = (t + 158) + offset_day * 24
            dow = ((this_day_08 % WEEK) // 24)
            is_weekday = dow < 5

            if is_weekday:
                day_booked = 0
                for h in range(this_day_08, this_day_08 + 10):
                    day_booked += self.admit_booked.get(h, 0)

                if day_booked <= 12:
                    intake_day = 2
                elif day_booked <= 24:
                    intake_day = 3
                else:
                    intake_day = 4

                for h in range(max(t + 14, this_day_08), min(this_day_08 + 10, t + 14 + WEEK)):
                    self.admit_quota[h] = max(self.admit_quota.get(h, 0), intake_day)
            else:
                for h in range(max(t + 14, this_day_08), min(this_day_08 + 10, t + 14 + WEEK)):
                    self.admit_quota[h] = max(self.admit_quota.get(h, 0), 1)

        # ---------- 3) Safety net for the rest of the "< 1 week" window ----------
        for h in range(t + 14, t + 14 + WEEK):
            hod = h % 24
            weekday = ((h % WEEK) // 24) < 5
            base = 2 if (weekday and 8 <= hod < 12) else 1
            self.admit_quota[h] = max(self.admit_quota.get(h, 0), base)

        return changes

    # ----------------- PLAN -----------------
    def plan(self, cases_to_plan, cases_to_replan, simulation_time: int):
        """
        LWF priority + capacity-aware admission with morning preference and load tie-breakers.
        Replanning allowed only when safe (>14d) and beneficial.
        
        FIXED: The signature now matches the Planner interface exactly:
        - cases_to_plan: list of case IDs
        - cases_to_replan: list of case IDs  
        - simulation_time: current simulation time
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
        Preference: preferred weekday mornings, then any hour; tie-breaker by load ratio.
        """
    def _choose_slot_improved(self, earliest_hour: int, prefer_mornings: bool = True):
        """
        אלגוריתם משופר: נעדיף שעות מוקדמות יותר עם רק מעט יותר עומס
        במקום שעות מאוחרות עם עומס נמוך
        """
        start = int(earliest_hour)
        end = start + LOOKAHEAD_HOURS
    
        best_h, best_score = None, 1e9
    
        for h in range(start, end):
            if self._has_slack(h):
                # ציון משולב: עומס + מרחק זמן (מעדיפים מוקדם יותר)
                load_ratio = self._slot_load_ratio(h)
                time_penalty = (h - start) / (24 * 7)  # עונש על זמן רחוק
            
                combined_score = load_ratio + 0.3 * time_penalty
            
            # בונוס לשעות מועדפות
                if self._is_preferred_time(h):
                    combined_score *= 0.7
                
                if combined_score < best_score:
                    best_h, best_score = h, combined_score
    
        return best_h

    def _slot_load_ratio(self, h: int):
        q = self.admit_quota.get(h, 0)
        b = self.admit_booked.get(h, 0)
        if q <= 0:
            return 1e9
        return b / q


# -------------------- Scoring helpers --------------------
def compute_normalized_scores(result, benchmark):
    """
    Compute normalized improvements (%) and final score with COST weighted x3.
    """
    norm_WTA = 100 * (result["waiting_time_for_admission"] - benchmark["waiting_time_for_admission"]) / benchmark["waiting_time_for_admission"]
    norm_WTH = 100 * (result["waiting_time_in_hospital"] - benchmark["waiting_time_in_hospital"]) / benchmark["waiting_time_in_hospital"]
    norm_NERV = 100 * (result["nervousness"] - benchmark["nervousness"]) / benchmark["nervousness"]
    norm_COST = 100 * (result["personnel_cost"] - benchmark["personnel_cost"]) / benchmark["personnel_cost"]
    final_score = (norm_WTA + norm_WTH + norm_NERV + 3 * norm_COST) / 6
    return {
        "norm_WTA": norm_WTA,
        "norm_WTH": norm_WTH,
        "norm_NERV": norm_NERV,
        "norm_COST": norm_COST,
        "final_score": final_score
    }


# -------------------- Main --------------------
if __name__ == "__main__":
    # Tip: for quick tests, use a shorter horizon (e.g., 24*7). For submission use 365*24.
    RUN_HOURS = 24 * 365  # 30 days for debugging, change to 365*24 for final submission

    problem = HealthcareProblem()
    planner = HeuristicPlanner("./temp/event_log.csv", ["diagnosis"])
    simulator = Simulator(planner, problem)

    result = simulator.run(RUN_HOURS)
    print("Raw results:", result)

    normalized = compute_normalized_scores(result, BENCHMARK_RESULTS)
    print("Normalized results (improvement %):", normalized)

    # Safe plot: use the same horizon you simulated
    end_idx = min(RUN_HOURS, len(planner.resource_reporter.available_resources))
    if end_idx > 0:
        planner.resource_reporter.create_graph(0, end_idx)