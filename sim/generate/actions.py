import torch
import mpm as us

from sim.util import dict_list_to_tensor
from sim.generate.ee import transform_by_quat


class ActionFactory:

    @staticmethod
    def get_goals(action_configs, overrides=None):
        if overrides is None:
            overrides = [None] * len(action_configs)
        goals_actions = []  # list of (sub)goals for all actions
        for action_config, override in zip(action_configs.values(), overrides):
            if override is not None:
                action_config.update(override)
            cfg = dict_list_to_tensor(action_config)
            action = ActionFactory.get_action(cfg)
            if hasattr(cfg, 'init_pos') and hasattr(cfg, 'init_quat'):
                action_init = action.get_init(cfg)
            else:
                action_init = None
            action_from = action.get_from(cfg)
            action_to = action.get_to(cfg)
            subgoals_action = action.get_goals(action_from, action_to, cfg, action_init)

            goals_actions += subgoals_action
        return goals_actions

    @staticmethod
    def get_action(config):
        if config.type == 'push':
            return Push
        elif config.type == 'grasp':
            return Grasp
        else:
            raise NotImplementedError(f"Action {config.type} not implemented.")

class Action:

    @staticmethod
    def get_goal(name, type, goal, wait_after):
        if type == 'grip':
            pass  # assert goal.shape[0] == 1
        elif type == 'move':
            assert goal.shape[0] in [3, 4, 7,]  # pos, quat, pos+quat
        elif type == 'set':
            assert goal.shape[0] in [7, 8,]  # pos+quat, pos+quat+open
        else:
            raise ValueError(f"Unknown goal type {type}.")
        return {'name': name, 'type': type, 'goal': goal, 'wait_after': wait_after}

    @staticmethod
    def get_goals(action_from, action_to, config, action_init=None):
        if action_init is not None:
            goals = [Action.get_goal('init', 'set', action_init, config.wait)]
            if action_from.shape[0] == 8:
                action_from = action_from[:7]  # initial gripper opening set by action_init instead
            goals += [Action.get_goal('from', 'move', action_from, config.wait)]
        else:  # use 'from' as initial state
            goals = [Action.get_goal('from', 'set', action_from, config.wait),]
        goals += [Action.get_goal('to', 'move', action_to, config.wait),]
        return goals
    
    @staticmethod
    def get_init(config):
        return torch.tensor([*config.init_pos, *config.init_quat, config.init_d]).cuda().to(us.FTYPE_TC)

    @staticmethod
    def get_from(config):
        raise NotImplementedError
    
    @staticmethod
    def get_to(config):
        raise NotImplementedError
    

class Push(Action):

    """
    Define where we want to end up. The start pose is defined by the offset and the end pose.
    """
    
    @staticmethod
    def get_from(config):
        from_pos = config.to_pos + transform_by_quat(config.from_offset, config.to_quat)
        return torch.tensor([*from_pos, *config.to_quat]).cuda().to(us.FTYPE_TC)
    
    @staticmethod
    def get_to(config):
        return torch.tensor([*config.to_pos, *config.to_quat]).cuda().to(us.FTYPE_TC)


class Grasp(Push):

    """
    Add a desired opening at start and end pose.
    """

    @staticmethod
    def get_goals(action_from, action_to, config, action_init=None):
        goals = Push.get_goals(action_from, action_to, config, action_init)
        goals += [Action.get_goal('close', 'grip', config.close_d, config.wait),]
        return goals

    @staticmethod
    def get_from(config):
        from_pose = Push.get_from(config)
        return torch.tensor([*from_pose, config.init_d]).cuda().to(us.FTYPE_TC)
