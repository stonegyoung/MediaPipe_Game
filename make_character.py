class SunState:
    def __init__(self):
        # 호랑이 손
        self.th_left_state = False
        self.th_right_state = False
        
        self.ms_state = False
        self.ms_cnt = 0
        
        self.tt_left_state = False
        self.tt_right_state = False
        self.tt_last_hand = ''
        self.tt_cnt = 0
        
    ### 게임 나가면 초기화
    def reset(self):
        self.th_left_state = False
        self.th_right_state = False
        self.ms_state = False
        self.ms_cnt = 0
        self.tt_left_state = False
        self.tt_right_state = False
        self.tt_last_hand = ''
        self.tt_cnt = 0
        
class MoonState:
    def __init__(self):
        # 호랑이 손
        self.th_left_state = False
        self.th_right_state = False
        
        self.ms_state = False
        self.ms_cnt = 0
        
        self.tt_left_state = False
        self.tt_right_state = False
        self.tt_last_hand = ''
        self.tt_cnt = 0
        
    ### 게임 나가면 초기화
    def reset(self):
        self.th_left_state = False
        self.th_right_state = False
        self.ms_state = False
        self.ms_cnt = 0
        self.tt_left_state = False
        self.tt_right_state = False
        self.tt_last_hand = ''
        self.tt_cnt = 0