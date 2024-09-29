import mpm as us
import torch


class StateMachine:

    def __init__(self, ee) -> None:
        self.status = 0
        self.ee = ee
    
    def reset(self):
        self.status = 0
        self.wait = 0
        self.ee.reset()
    
    def get_goal(self):
        return self.ee.get_goal()

    def set_goal(self, goal):
        if goal is not None:
            goal_state = self.ee.get_state()
            if goal['type'] == 'set':
                # directly override state - no additional movement needed
                if self.ee.is_gripper and len(goal['goal']) == 7:
                    goal['goal'] = torch.tensor([*goal['goal'], goal_state[7]]).cuda().to(us.FTYPE_TC)  # use current as default
                self.ee.set_state(goal['goal'])
                goal_state = goal['goal']  # for consistency
            elif goal['type'] == 'move':
                pose = goal['goal']
                if pose.shape[0] == 7:  # pos + quat
                    goal_state[:7] = pose.clone()
                elif pose.shape[0] == 4:  # just quat
                    goal_state[3:7] = pose.clone()
                elif pose.shape[0] == 3:  # just pos
                    goal_state[:3] = pose.clone()
            elif goal['type'] == 'grip':  # just opening
                assert self.ee.is_gripper
                goal_state[7] = goal['goal']
            else:
                raise ValueError(f"unknown goal type {goal['type']}")
        else:
            goal_state = None
        self.ee.set_goal(goal_state)

    def step(self, goals):
        if self.wait > 0:
            self.wait -= 1
            return False
        
        for status, goal in enumerate(goals):
            if self.status == status:  # our current goal
                if self.get_goal() is None:
                    us.logger.debug(f"setting new goal {status}: {goal['name']}")
                    self.set_goal(goal)
                    goal_reached = False
                else:
                    goal_reached = self.ee.step()

                if goal_reached:
                    us.logger.debug(f"arrived at goal {status}: {goal['name']}")
                    self.set_goal(None)  # -> controller outputs zero velocities
                    if 'wait_after' in goal:
                        self.wait = goal['wait_after']
                    self.status += 1
                else:
                    return False
        us.logger.debug("all goals done")
        return True
