from .trip_acc_egr_choice import combine_mode_ids
from itertools import product
import pandas as pd

MODE_INVALID = 0
MODE_WALK = 1
MODE_BIKE = 2
MODE_EBIKE = 3
MODE_CAR = 4
MODE_CP = 5
MODE_DRT = 6
MODE_PT = 7

# TRIP TYPE
OUTBOUND_TRIP = 1
INBOUND_TRIP = 2
INTERMEDIATE_TRIP = 3
FIELD_AUTO_OWNERSHIP = 'has_car'
FIELD_BIKE_OWNERSHIP = 'has_bike'
FIELD_EBIKE_OWNERSHIP = 'has_ebike'
FIELD_DRIVING_LICENCE = 'driving_license'
FIELD_MAAS_SUBSCRIPTION = 'maas_subscription'


class PersonUtils:
    @staticmethod
    def mode_availability(person, mode):
        """
        check if person owns the required mode resource. if this is a MaaS mode, check if the agent has a subscription
        :param person
        :param mode: TrafficMode
        :return: True if this person can use this mobility, otherwise False
        """
        if not mode.is_multi_modal:
            if mode.main_mode == MODE_CAR:
                return person[FIELD_DRIVING_LICENCE] and person[FIELD_AUTO_OWNERSHIP]
            elif mode.main_mode in [MODE_BIKE]:
                return person[FIELD_BIKE_OWNERSHIP]
            elif mode.main_mode in [MODE_EBIKE]:
                return person[FIELD_EBIKE_OWNERSHIP]
            return True
        else:
            result = True
            if mode.main_mode == MODE_CAR:
                result = result and person[FIELD_DRIVING_LICENCE] and person[FIELD_AUTO_OWNERSHIP]

            if mode.access_mode == MODE_CAR or mode.egress_mode == MODE_CAR:
                result = result and person[FIELD_DRIVING_LICENCE] and person[FIELD_AUTO_OWNERSHIP]

            # check the bike/ebike ownership, they are not exclusive, e.g. ebike_pt_bike, require both ownership
            if mode.access_mode == MODE_BIKE or mode.egress_mode == MODE_BIKE:
                result = result and person[FIELD_BIKE_OWNERSHIP]
            if mode.access_mode == MODE_EBIKE or mode.egress_mode == MODE_EBIKE:
                result = result and person[FIELD_EBIKE_OWNERSHIP]
            return result

    @staticmethod
    def MaaS_subscription_available(person, mode):
        """
        The mode is available all the time, no ownership is required, but the person should have the right MaaS
        subscription, e.g. subscription for car-share, subscription for bike-share, etc.
        :param person:
        :param mode:
        :return:
        """
        if mode.main_mode == MODE_CAR:
            return person[FIELD_DRIVING_LICENCE] and person[FIELD_MAAS_SUBSCRIPTION]
        elif mode.main_mode in [MODE_BIKE]:
            return person[FIELD_MAAS_SUBSCRIPTION]
        elif mode.main_mode in [MODE_EBIKE]:
            return person[FIELD_MAAS_SUBSCRIPTION]
        elif mode.main_mode in [MODE_DRT]:
            return person[FIELD_MAAS_SUBSCRIPTION]
        elif mode.main_mode in [MODE_PT]:
            return person[FIELD_MAAS_SUBSCRIPTION]
        elif mode.is_multi_modal:
            if mode.access_mode == MODE_DRT or mode.egress_mode == MODE_DRT:
                return person[FIELD_MAAS_SUBSCRIPTION]
            if mode.access_mode == MODE_PT or mode.egress_mode == MODE_PT:
                return person[FIELD_MAAS_SUBSCRIPTION]
            if mode.access_mode == MODE_CAR or mode.egress_mode == MODE_CAR:
                return person[FIELD_DRIVING_LICENCE] and person[FIELD_MAAS_SUBSCRIPTION]
            # check the bike/ebike subscription, they are not exclusive, e.g. ebike_pt_bike, require both subscription
            result = True
            if mode.access_mode == MODE_BIKE or mode.egress_mode == MODE_BIKE:
                result = person[FIELD_MAAS_SUBSCRIPTION]
            if mode.access_mode == MODE_EBIKE or mode.egress_mode == MODE_EBIKE:
                result = result and person[FIELD_MAAS_SUBSCRIPTION]
            return result
        else:
            return True


class TrafficMode:
    mode_name_map = {'-': MODE_INVALID, 'car': MODE_CAR, 'cp': MODE_CP, 'bike': MODE_BIKE, 'ebike': MODE_EBIKE, 'drt': MODE_DRT,
                     'walk': MODE_WALK, 'pt': MODE_PT}
    mode_id_name_map = {val: key for key, val in mode_name_map.items()}

    def __init__(self, **kwargs):
        # self.lvl_1_name__ = kwargs['L1'].lower()
        # self.lvl_2_name__ = kwargs['L2'].lower()

        # expected input mode format is:  access_main_egress
        self.lvl_3_name__ = kwargs['NAME'].lower()

        # indicate whether the mode require MaaS subscription
        # determine whether this mode is a MaaS mode: it requires the "MaaS subscription"
        # self.MaaS_subscription_required__ = kwargs[FIELD_MAAS_SUBSCRIPTION]

        """
        # "use_shared_mode" is just an input, it indicates that whether to use a shared mode or not when an ownership
        # is required       
         Important: when dealing with mode walk_car_bike, we either consider both modes use private owned car and bike,
                    or the car-share or bike-share. This depends on the use_shared_mode argument
                    if the input mode is a multi-modal, we split it into access + main + egress
        """
        # set it always to be False, whether to use share is depends on the personal attribute, not mode itself!
        self.use_shared_mode__ = kwargs['use_shared_mode'] # and kwargs['maas_subscription']

        # if the mode is "CAR" and use_shared_mode is TRUE, it means that it is a CARShr mode which does not
        # require vehicle ownership
        # if self.own_mode_required__ and self.use_shared_mode__:
        #     self.own_mode_required__ = False

        '''A. get the access mode, main mode and egress mode for multi-modal'''
        pos = self.lvl_3_name__.find('_')
        pos_next = self.lvl_3_name__.rfind('_')
        if pos_next > pos > 0:
            self.access_mode__ = self.mode_name_map[self.lvl_3_name__[0: pos]]
            self.egress_mode__ = self.mode_name_map[self.lvl_3_name__[pos_next + 1:]]
            self.main_mode__ = self.mode_name_map[self.lvl_3_name__[pos+1: pos_next]]

            '''B. check if the mode requires ownership, taking the use_shared_mode into account'''
            if self.access_mode__ in [MODE_CP, MODE_DRT, MODE_WALK, MODE_PT]:
                self.access_mode_requires_ownership__ = int(False)
            else:
                self.access_mode_requires_ownership__ = int(not self.use_shared_mode__)

            if self.egress_mode__ in [MODE_CP, MODE_DRT, MODE_WALK, MODE_PT]:
                self.egress_mode_requires_ownership__ =  int(False)
            else:
                self.egress_mode_requires_ownership__ = int(not self.use_shared_mode__)
        else:
            self.access_mode__ = MODE_INVALID
            self.egress_mode__ = MODE_INVALID
            self.main_mode__ = self.mode_name_map[self.lvl_3_name__]
            self.access_mode_requires_ownership__ = int(False)
            self.egress_mode_requires_ownership__ = int(False)

        if self.main_mode__ not in [MODE_CP, MODE_DRT, MODE_WALK, MODE_PT]:
            self.main_mode_requires_ownership__ = int(not self.use_shared_mode__)
        else:
            self.main_mode_requires_ownership__ = int(False)

    @property
    def requires_bike_or_ebike(self):
        """
        Check if the mode requires a bike or ebike in access, main or egress
        :return:
        """
        if self.access_mode__ in (MODE_BIKE, MODE_EBIKE) or self.egress_mode__ in (MODE_BIKE, MODE_EBIKE) \
                or self.main_mode__ in (MODE_BIKE, MODE_EBIKE):
            return True
        else:
            return False

    @property
    def mode_name(self):
        return self.lvl_3_name__

    def is_multi_modal_valid(self, trip_type):
        """
        For multi-modal and use_share_mode=False: Determine if this multi-modal mode is a valid combination or not,
         e.g. for outbound trip: bike-car-walk is not a valid mode
        :return:
        """
        if not self.is_multi_modal:
            return False

        # always true if sharing is allowed (scenario input)
        if self.use_shared_mode__:
            return True

        '''for outbound trip: 
            1.	Walk_CAR_Bike is valid (bike in the hub/station)
            2.	(e)Bike_car_walk is not valid;
            3.	(e)Bike_CP_ * is valid?
            4.	CAR can’t be an egress mode (e.g. walk-bike-car, where is the car from?);
            5.	(e)Bike can be an egress mode (bike in a hub/station);
            6.	PT + CP + walk is valid
        '''
        if trip_type == OUTBOUND_TRIP:
            if self.access_mode__ in (MODE_BIKE, MODE_EBIKE, MODE_PT, MODE_DRT, MODE_CP) and self.main_mode__ == MODE_CAR:
                return False
            if self.main_mode__ in (MODE_BIKE, MODE_EBIKE) and self.egress_mode__ == MODE_CAR:
                return False
            if self.egress_mode__ == MODE_CAR:
                return False
        elif trip_type == INBOUND_TRIP:
            '''for inbound trip: 
                1. CAR + (e)bike is not valid
                2. CAR can't be an access mode in a multi-modal mode
                3. (e)bike can be an access mode (put it in a hub/station)    
            '''
            if self.access_mode__ == MODE_CAR and self.main_mode__ in (MODE_BIKE, MODE_EBIKE):
                return False
            if self.main_mode__ == MODE_CAR and self.egress_mode__ in (MODE_BIKE, MODE_EBIKE):
                return False
        else:
            return True
        return True

    @property
    def is_multi_modal(self):
        return self.access_mode__ != MODE_INVALID or self.egress_mode__ != MODE_INVALID

    @property
    def main_mode(self):
        return self.main_mode__

    @property
    def access_mode_ori(self):
        return self.access_mode__

    @property
    def access_mode(self):
        if self.access_mode__ == MODE_WALK:
            return self.main_mode__
        else:
            return self.access_mode__

    @property
    def egress_mode_ori(self):
        return self.egress_mode__

    @property
    def egress_mode(self):
        if self.egress_mode__ == MODE_WALK:
            return self.main_mode__
        else:
            return self.egress_mode__

    @property
    def main_requires_mode_ownership(self):
        return bool(self.main_mode_requires_ownership__)

    @property
    def access_requires_mode_ownership(self):
        """
        if access mode is Walk, we check the main mode's ownership
        :return:
        """
        try:
            if self.access_mode__ == MODE_WALK:
                return bool(self.main_mode_requires_ownership__)
            else:
                return bool(self.access_mode_requires_ownership__)
        except ValueError:
            print("error")

    @property
    def egress_requires_mode_ownership(self):
        if self.egress_mode__ == MODE_WALK:
            return bool(self.main_mode_requires_ownership__)
        else:
            return bool(self.egress_mode_requires_ownership__)

    @property
    def required_num_of_ownerships(self):
        return self.access_mode_requires_ownership__ + self.egress_mode_requires_ownership__ + self.main_mode_requires_ownership__

    @property
    def requires_only_1_ownerships(self):
        """
        if the outbound trip mode is: bike-pt-walk, the matched inbound trip mode is (* but not bike)-(* but not bike)-bike;
        if the outbound trip mode is: walk-car-bike, the matched inbound trip mode is bike-car-*;
        :return:
        """
        return self.required_num_of_ownerships == 1

    @property
    def requires_at_least_2_ownerships(self):
        """
        if the outbound trip mode is: bike-pt-bike, the matched inbound trip mode is bike-*-bike;
        if the outbound trip mode is: walk-car-bike, the matched inbound trip mode is bike-car-*;
        :return:
        """
        # at least 2 modes requires ownership
        return self.required_num_of_ownerships >= 2

    def is_matched_outbound_mode(self, owned_mode_at_origin, outbound_mode):
        """
        This function is only used by inbound trip when the multi-modal self is valid;
        :param owned_mode_at_origin:
        :param outbound_mode:
        :return:
        """
        '''1. check candidate's ACCESS mode'''
        # owned mode is NOT at origin, but candidate's access requires mode ownership, not OK
        if owned_mode_at_origin is None and self.access_requires_mode_ownership:
            return False
        # owned mode is at origin, candidate's access mode does not require it, not OK
        if owned_mode_at_origin is not None and self.access_mode__ != owned_mode_at_origin:
            return False

        '''2. outbound's main requires ownership, the candidate's MAIN should be the same as outbound's main 
        e.g. walk-car-bike, bike-car-walk'''
        if outbound_mode.main_requires_mode_ownership:
            if self.main_mode__ != outbound_mode.main_mode:
                return False

        '''3. check the candidate's EGRESS mode compatible with outbound mode's access mode'''
        # outbound access need ownership but inbound egress does not or another way around, not OK
        if outbound_mode.access_requires_mode_ownership != self.egress_requires_mode_ownership:
            return False
        # if both need ownership, then the mode name should be the same
        elif outbound_mode.access_requires_mode_ownership and self.egress_requires_mode_ownership:
            # should be the SAME mode
            if self.egress_mode != outbound_mode.access_mode:
                return False
        return True

    # @property
    # def is_maas_mode(self):
    #     """
    #     :return: true if it is a MaaS mode
    #     """
    #     if self.is_multi_modal:
    #         return self.use_shared_mode__
    #     else:
    #         if self.main_mode__ == MODE_WALK:
    #             return False

    @property
    def use_share_mode(self):
        return self.use_shared_mode__


def check_constraints(candidate_mode, person, is_inbound_trip, is_outbound_trip, outbound_mode, owned_mode_at_trip_origin, owned_mode_name_at_trip_origin, is_outbound_of_a_subtour):
    """
    check the contraints on trips mode in a tour. (No MaaS mode)
    :param candidate_mode:
    :param person:
    :param is_inbound_trip:
    :param is_outbound_trip:
    :param outbound_mode:
    :param owned_mode_at_trip_origin:
    :param owned_mode_name_at_trip_origin:
    :param is_outbound_of_a_subtour: it is the outbound trip of a sub tour
    :return:
    """
    rtn_outbound_mode_name = outbound_mode
    constraints_satisfied = True

    # # when all mode don't require ownership (e.g. Car is actually CarShr, bike is bikeShr)
    # if candidate_mode.use_share_mode:
    #     if PersonUtils.MaaS_subscription_available(person, candidate_mode):
    #         return True, rtn_outbound_mode_name
    #     else:
    #         return False, rtn_outbound_mode_name

    '''constraint 0: agent is capable to use this mode, mode OWNERSHIP or MaaS'''
    if not PersonUtils.mode_availability(person, candidate_mode):
        return False, rtn_outbound_mode_name

    # constraint 1. Is candidate mode AVAILABLE at the trip's origin area?
    # todo constraint 2. Is candidate mode ALLOWED in the trip's destination area?
    # todo constraint 3. Is candidate mode suitable for this trip qua distance and available time?

    '''constraint 4: Is candidate mode CONSISTENT with previous trips' mode?'''
    # 1. for inbound trip
    if is_inbound_trip:
        # 1.a. for multi-modal mode
        if candidate_mode.is_multi_modal:
            # CAR can't be used as access mode in a inbound trip (checked by is_multi_modal_valid function)
            if not candidate_mode.is_multi_modal_valid(INBOUND_TRIP):
                constraints_satisfied = False
            else:
                # if owned mode is at trip origin, we should bring it back, only (e)bike-*-* or bike-car-* are allowed
                if owned_mode_at_trip_origin:
                    constraints_satisfied = candidate_mode.is_matched_outbound_mode(owned_mode_name_at_trip_origin, outbound_mode)
                # when owned mode is NOT at trip origin,
                else:
                    constraints_satisfied = candidate_mode.is_matched_outbound_mode(None, outbound_mode)
        # 1.b. for uni-modal mode
        else:
            # when owned mode is at the trip origin,
            if owned_mode_at_trip_origin:
                if candidate_mode.main_requires_mode_ownership:
                    # when owned mode is here, candidate mode requires it, compare the mode name
                    if candidate_mode.main_mode != owned_mode_name_at_trip_origin:
                        constraints_satisfied = False
                    # to avoid walk-car-bike + bike: e.g outbound mode: walk-car-bike, own_mode at location is bike,
                    # valid inbound mode is bike-car-walk, bike as unimodal is NOT valid
                    if outbound_mode.requires_at_least_2_ownerships:
                        constraints_satisfied = False
                    # if outbound_mode.access_mode is not None:
                    #     if candidate_mode.mode_name != outbound_mode.pt_access_mode:
                    #         constraints_satisfied = False
                else:
                    # when owned mode is at trip origin, the target mode doesn't required to own, it is no allowed
                    constraints_satisfied = False
            # when owned mode is NOT at the trip origin,
            else:
                # but candidate mode(unimodal here) requires ownership, it is not possible.
                if candidate_mode.main_requires_mode_ownership:
                    constraints_satisfied = False
                # but outbound trip mode requires ownership and we know the own mode is not at trip origin, it means
                # the own mode requirement must be in the main or egress mode, which is NOT possible for this unimodal
                # candidate mode
                elif outbound_mode.requires_only_1_ownerships or outbound_mode.requires_at_least_2_ownerships:
                    constraints_satisfied = False
    # 2. for outbound trip, which means the agent will come back to the trip's origin later in the tour
    elif is_outbound_trip:
        # 2.a. for multi-modal mode
        if candidate_mode.is_multi_modal:
            if not candidate_mode.is_multi_modal_valid(OUTBOUND_TRIP):
                constraints_satisfied = False
            # if candidate mode's egress mode need ownership, we are sure the egress mode is bike, because the
            # "valid multi-modal" process controls that

            if is_outbound_of_a_subtour:
                # special case: in a sub-tour, if in the outbound of the tour with CAR, then in the sub-tour, Bike
                # can't be used in access, main or egress
                if candidate_mode.requires_bike_or_ebike and (owned_mode_name_at_trip_origin is None or owned_mode_name_at_trip_origin == MODE_CAR):
                    constraints_satisfied = False
                # in a sub-tour, Bike can't be used in egress
                if candidate_mode.egress_requires_mode_ownership and (owned_mode_name_at_trip_origin is None or owned_mode_name_at_trip_origin in (MODE_BIKE, MODE_EBIKE)):
                    constraints_satisfied = False

            # check if candidate access mode requires mode ownership,
            if candidate_mode.access_requires_mode_ownership:
                # but owned mode is not here, not possible
                if not owned_mode_at_trip_origin:
                    constraints_satisfied = False
                else:
                    # when owned mode is available, check if their name are the same. if owned mode name at trip
                    # origin is None, it is the first trip, so 'Car-PT-Walk' mode is possible
                    if owned_mode_name_at_trip_origin is not None:
                        if candidate_mode.access_mode != owned_mode_name_at_trip_origin:
                            constraints_satisfied = False
            # store the outbound's mode name
            if constraints_satisfied:
                rtn_outbound_mode_name = candidate_mode
        # 2.b. for uni-modal mode
        else:
            # when the owned mode is at trip origin,
            if owned_mode_at_trip_origin:
                # and candidate mode requires to own mode. E.g for sub-tour origin: if car available, check if the name of candidate mode's name is car
                if candidate_mode.main_requires_mode_ownership:
                    # if owned mode name at trip origin is None, means it is the first trip of whole tour
                    if owned_mode_name_at_trip_origin is not None:
                        if candidate_mode.main_mode != owned_mode_name_at_trip_origin:
                            constraints_satisfied = False
            # when the owned mode is NOT at trip origin:
            else:
                # but the candidate mode requires ownership, it is not possible
                if candidate_mode.main_requires_mode_ownership:
                    constraints_satisfied = False

            # store the outbound's mode name
            if constraints_satisfied:
                rtn_outbound_mode_name = candidate_mode

    # 3. intermediate trip (agent won't come back to this trip origin)
    else:
        # 3.a. MULTI-MODAL mode
        if candidate_mode.is_multi_modal:
            # when owned mode is at trip origin, agent should always take it, non PT possible except bike-PT-bike
            if owned_mode_at_trip_origin:
                if not ((candidate_mode.access_mode == owned_mode_name_at_trip_origin == MODE_BIKE or
                        candidate_mode.access_mode == owned_mode_name_at_trip_origin == MODE_EBIKE) and
                        candidate_mode.egress_mode in (MODE_BIKE, MODE_EBIKE)):
                    constraints_satisfied = False
            # when owned mode is NOT at trip origin,
            else:
                # the candidate access or egress mode should NOT require mode ownership
                if candidate_mode.access_requires_mode_ownership or candidate_mode.egress_requires_mode_ownership:
                    constraints_satisfied = False
        # 3.b. UNI-MODAL mode
        else:
            # when own mode is at trip origin,
            if owned_mode_at_trip_origin:
                # the candidate mode should own the mode in order to take it away
                if not candidate_mode.main_requires_mode_ownership:
                    constraints_satisfied = False
                else:
                    # candidate mode requires ownership, compare the name with the owned name at trip origin, because
                    # the owned mode is here, the owned mode name should NOT be None
                    if candidate_mode.main_mode != owned_mode_name_at_trip_origin:
                        constraints_satisfied = False
            # when owned mode is NOT at trip origin,
            else:
                # if mode is a MaaS mode, we can use it. Otherwise, we need to check further,
                # if not candidate_mode.is_maas_mode:
                # if candidate mode requires mode ownership, not possible
                if candidate_mode.main_requires_mode_ownership:
                    constraints_satisfied = False

    return constraints_satisfied, rtn_outbound_mode_name


# store the outbound trip's origin and mode, they are: tour start location, sub-tour start location
outbound_trip_origin_and_mode = {}


def generate_tour_modes_options(trip_index, person, trips, modes, my_own_mode_name, outbound_mode,
                                my_own_mode_location, tour_modes, trip_modes_sets):
    """
    take the trips of a tour of a person into consideration, find all the possible modes combinations
    :param trip_index: index of the trip in a tour
    :param person: a person's information
    :param trips: a person's single tour (several trips)
    :param modes: all allowed modes
    :param my_own_mode_name: owned mode name, e.g. 'private car' or 'bike' which owns by the agent
    :param outbound_mode: traffic mode instance
    :param my_own_mode_location:  location of the person own mode(car/bike)
    :param tour_modes: output container
    :param trip_modes_sets: output
    :return:
    """
    """
    python pass by reference, we don't want to the value be changed by recursive call, we need to keep the function
    argument unchanged
    """
    # local_other_modes_allow_to_use = copy(other_modes_allow_to_use)
    # local_my_own_mode_location = copy(my_own_mode_location)  # own mode location
    # local_my_own_mode_name = copy(my_own_mode_name)          # own mode name at that location
    local_my_own_mode_location = my_own_mode_location  # own mode location
    local_my_own_mode_name = my_own_mode_name  # own mode name at that location
    local_outbound_mode = outbound_mode

    # determine the trip type: outbound, inbound, or intermediate trip
    curt_trip_origin = trips.iloc[trip_index].origin
    curt_trip_dest = trips.iloc[trip_index].destination

    # check if the trip is outbound or not
    is_outbound_trip = False
    is_outbound_of_a_subtour = False
    if trip_index == 0:
        outbound_trip_origin_and_mode.clear()
        # first trip is always an outbound
        outbound_trip_origin_and_mode[curt_trip_origin] = None
        is_outbound_trip = True
    else:
        '''
        check if previous trip an interzone-trip, why?
          trip  From To
          1     45   45  (interzone trip)   outbound trip
          2     45   93                     NOT outbound, but intermediate trip!!
          3     93   45
        '''
        # if agent will return to this trip's origin, then this trip is an outbound trip
        is_previous_trip_interzone = trips.iloc[trip_index-1].origin == trips.iloc[trip_index-1].destination
        if (not is_previous_trip_interzone) and (curt_trip_origin in trips['destination'].iloc[trip_index+1:].tolist()):
            outbound_trip_origin_and_mode[curt_trip_origin] = None
            is_outbound_trip = True
            is_outbound_of_a_subtour = True

    # check if the trip is inbound: if the trip destination is the outbound trip's origin, it is an inbound trip
    is_inbound_trip = False
    # make sure it is not an outbound trip, why this check? if origin zone=destination zone, without
    # (not is_outbound_trip), this trip is an outbound trip and also an inbound_trip
    if not is_outbound_trip and (curt_trip_dest in outbound_trip_origin_and_mode):
        is_inbound_trip = True
        # for inbound trip, get the corresponding outbound mode name
        local_outbound_mode = outbound_trip_origin_and_mode[curt_trip_dest]

    # loop through each mode as candidate mode
    for mode_id, mode in modes.items():

        tmp = curt_trip_origin == local_my_own_mode_location
        # if there is no own mode, the owned_mode_at_trip_origin is False. This extra check is due to interzone
        if trip_index > 0:
            tmp = (curt_trip_origin == local_my_own_mode_location and local_my_own_mode_name is not None)

        constraint_satisfied, local_outbound_mode = \
            check_constraints(candidate_mode=mode,
                              person=person,
                              is_inbound_trip=is_inbound_trip,
                              is_outbound_trip=is_outbound_trip,
                              outbound_mode=local_outbound_mode,
                              owned_mode_at_trip_origin=tmp,
                              owned_mode_name_at_trip_origin=local_my_own_mode_name,
                              is_outbound_of_a_subtour=is_outbound_of_a_subtour
                              )

        if not constraint_satisfied:
            continue

        # save the outbound mode
        if is_outbound_trip:
            outbound_trip_origin_and_mode[trips.iloc[trip_index].origin] = local_outbound_mode

        # save the trip's mode name in this tour, first 2 values are agent id and tour id
        # tour_modes[trip_index] = mode_id # mode.mode_name

        tour_modes[trip_index] = combine_mode_ids(mode.access_mode_ori, mode.main_mode, mode.egress_mode_ori)
        if trip_index == len(trips) - 1:
            """
            save the mode combination
            """
            # save
            trip_modes_sets.extend(tour_modes)
            continue

        """
        if current trip is with OWN mode(private car, bike), we keep track the OWN mode location, because the agent
        needs finally take them away. So we check if this person will return to this location in the 
        following trips. If not, the person can ONLY use its own mode. If yes, all the following trips include the
        trip which the person back to this location can use other modes. 
        example 1 trips:
           1 - 2
           2 - 3
           3 - 2
           2 - 4
           4 - 2
           2 - 1   
               After first trip with "car", the "car" is at location 2. since the person will return to this place,
           the trips 2-3, 3-2 may use all OTHER allowed modes.
               If the person with "car" for trip 2 - 3, the "car" is at location 3, since the person won't be back to
           this location, the person can ONLY use "car" in the trips after trip '2 - 3'.

        example 2 trips
           1 - 2
           2 - 3
           3 - 4
           4 - 1
           after first trip with "car", the "car" is at location 2, the person will NOT return to this place, so
           all the following trips the person can ONLY use "car"  

        For MUTLIMODAL with access and egress mode it is a bit different, with bike/car - * - Walk, the own mode
        will stay at the location nearby(5km radius?) ORIGIN, it means the trip to this ORIGIN must ONLY use 
        walk-PT-bike/car mode. 
        """
        # store the owned mode location and name if the selected mode requires ownership
        if not mode.is_multi_modal and mode.main_requires_mode_ownership:
            local_my_own_mode_name = mode.main_mode
            local_my_own_mode_location = trips.iloc[trip_index].destination
        elif mode.is_multi_modal and mode.egress_mode in (MODE_CAR, MODE_BIKE, MODE_EBIKE):
            local_my_own_mode_name = mode.egress_mode
            local_my_own_mode_location = trips.iloc[trip_index].destination

        generate_tour_modes_options(trip_index + 1, person, trips, modes, local_my_own_mode_name, local_outbound_mode,
                                    local_my_own_mode_location, tour_modes, trip_modes_sets)

        # reset
        # local_other_modes_allow_to_use = other_modes_allow_to_use
        local_my_own_mode_location = my_own_mode_location
        local_my_own_mode_name = my_own_mode_name
        local_outbound_mode = outbound_mode
        # local_required_PT_egress_mode = required_PT_egress_mode


def generate_maas_combination(num_trips, col_names, mode_list):
    """
    Since when MaaS is available, agent can use any multimodal in a trip, so any multimodal combination in a TOUR.
    So just simply create all combinations of the modes for each type of tour
    :param num_trips:
    :param col_names:
    :param mode_list:
    :return:
    """
    return pd.DataFrame(list(product(mode_list, repeat=num_trips)), columns=col_names)
