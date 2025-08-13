from simulator import Simulator
from planners import Planner
from problems import HealthcareProblem, ResourceType
from reporter import EventLogReporter, ResourceScheduleReporter


class NaivePlanner(Planner):
    def __init__(self, eventlog_file, data_columns):
        super().__init__()
        self.eventlog_reporter = EventLogReporter(eventlog_file, data_columns)
        self.resource_reporter = ResourceScheduleReporter()
        self.replanned_patients = set()

    def report(self, case_id, element, timestamp, resource, lifecycle_state, data=None):
        '''
        Each time a simulation event happens, this method is invoked with the following parameters:
        - case_id: the case id of the patient to which the event applies
        - element: the process element (task or event) that started or completed (if any)
        - timestamp: the simulation time at which the event happened
        - resource: the resource that the patient is using (if any)
        - lifecycle_state: the lifecycle state of the element ("start", "complete")
        - data: a dictionary with additional data for the event (if any)
        You can choose to use or store this information or simply ignore it (i.e. add a 'pass' statement here).
        Example uses are to store the event in an event log to be used for process mining later as per the example below.
        '''
        self.eventlog_reporter.callback(case_id, element, timestamp, resource, lifecycle_state)
        self.resource_reporter.callback(case_id, element, timestamp, resource, lifecycle_state, data)

    def plan(self, cases_to_plan, cases_to_replan, simulation_time):
        '''
        Each time a new task is enabled or a resource becomes available, this method is invoked so you can decide which patients to admit when.
        The method is invoked with the following parameters:
        - cases_to_plan: a list of case ids that can be planned for admission
        - cases_to_replan: a list of case ids that were already planned, but can still be replanned for admission
        - simulation_time: the current simulation time
        You must return a list of tuples (<case_id>, <time>) that represent at which time which patient should be admitted.
        You do not have to plan all patients. You can choose to plan none, some or all of them.
        Your plan must observe the following constraints:
        - patients must be planned for admission at least 24 hours ahead of their admission time.
        '''
        # We use a pretty naive approach for demonstration purposes:
        # - We initially plan all patients for 48 hours from now.
        # - We replan all patients that are already planned for 24 hours from now.
        planned_cases = []
        next_plannable_time = round(simulation_time + 48)
        next_replannable_time = round((simulation_time + 24) + 0.5)
        for case_id in cases_to_plan:
            planned_cases.append((case_id, next_plannable_time))
        for case_id in cases_to_replan:
            if case_id not in self.replanned_patients:
                planned_cases.append((case_id, next_replannable_time))
                self.replanned_patients.add(case_id)
        return planned_cases
    
    def schedule(self, simulation_time):
        '''
        Each day at 18:00 (in simulation time), resources can be scheduled. At that time this method is invoked to that end with the current simulation time.
        There are five resource types: ResourceType.OR, ResourceType.A_BED, ResourceType.B_BED, ResourceType.INTAKE, ResourceType.ER_PRACTITIONER
        These have maximum numbers of resources available (5, 30, 40, 4, 9).
        You must return a list of tuples (<type>, <time>, <number>). Each tuple should contain:
        - one of the five resource types.
        - the number of resources of that type that should be available.
        - the moment in (simulation) time from which that number of resources should be available.

        Note that a week has 168 hours and the simulation starts at 0, which is Monday 2018-01-01 00:00:00.000.
        Your schedule must observe the following constraints:
        - You must schedule for the start of tomorrow's working day or later (i.e. at least 14 hours ahead).
        - You must not schedule more resources than the maximum number of resources of a type.
        - If you are scheduling less than one week ahead on the same day (i.e., less than 158 hours ahead), you can only increase - not decrease - the number of resources.
        :return: [(ResourceType, simulation time in hours, number of resources), ...]
        '''
        # This is how you can get some interesting information from the timestamp
        hour_of_week = simulation_time % 168
        day_of_week = hour_of_week // 24  # Monday is 0, Tuesday is 1, ..., Sunday is 6
        is_weekday = day_of_week < 5
        hour_of_day = hour_of_week % 24  # Note that this is always 18:00, because that is the moment we plan

        # This is a simple example. We will schedule 1 week ahead.
        # On weekdays, we will schedule all resources for 8:00. For 18:00 we will schedule all beds and ER practitioners, but just 1 OR and 1 INTAKE.
        # On weekends, we do not change the schedule, which means that it will just have the resources from Friday 18:00 (all beds and ER practitioners, but just 1 OR and 1 INTAKE).
        if is_weekday:
            return [
                (ResourceType.OR, simulation_time + 158, 5),  # This day next week at 8:00, there will be 5 ORs
                (ResourceType.A_BED, simulation_time + 158, 30),  # This day next week at 8:00, there will be 30 A beds (this is superfluous, because this number never changes)
                (ResourceType.B_BED, simulation_time + 158, 40),  # ...
                (ResourceType.INTAKE, simulation_time + 158, 4),
                (ResourceType.ER_PRACTITIONER, simulation_time + 158, 9),
                (ResourceType.OR, simulation_time + 168, 1),  # This day next week at 18:00, there will be 1 OR
                (ResourceType.INTAKE, simulation_time + 168, 1)  # ...
            ]
        else:
            return []  # On weekends, we do not change the schedule.


if __name__ == '__main__':
    planner = NaivePlanner("./temp/event_log.csv", ["diagnosis"])
    problem = HealthcareProblem()
    simulator = Simulator(planner, problem)
    result = simulator.run(365*24)

    print(result)

    planner.resource_reporter.create_graph(168, 168*4)