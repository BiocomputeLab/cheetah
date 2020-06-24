
import time


def empty_analyse (t_step):
    '''Empty analyse function (does nothing)'''
    return None


def empty_control_action (t_step, data):
    '''Empty control action function (does nothing)'''
    return None


class ControlAlgorithm():
    '''Class for implementing a control algorithm'''


    def __init__ (self, analyse_fn=empty_analyse, 
                  control_action_fn=empty_control_action,
                  sleep_time=1.0):
        '''Initialization'''
        self.analyse_fn = analyse_fn
        self.control_action_fn = control_action_fn
        self.stop = False
        self.sleep_time = sleep_time # seconds
    

    def start (self, max_t_step=10000):
        t_step = 0
        while self.stop == False and t_step <= max_t_step:
            data = self.analyse_fn(t_step)
            self.control_action_fn(t_step, data)
            t_step += 1
            time.sleep(self.sleep_time)
