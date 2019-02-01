from . import neuralweb as nw
import pygame as pg
import time as t

class AIPaddle:
    def __init__(self, screen_rect, ball_rect, difficulty):
        self.difficulty = difficulty
        self.screen_rect = screen_rect
        self.ball_Rect = ball_rect
        self.move_up = False
        self.move_down = False
        self.screen_response_area_rect = self.screen_rect
        
        if self.difficulty == 'hard':
            num = 1
        elif self.difficulty == 'medium':
            num = 2
        elif self.difficulty == 'easy':
            num = 3
            
        surf = pg.Surface([self.screen_rect.width / num, self.screen_rect.height])
        self.screen_response_area_rect = surf.get_rect()
        
    def update(self, ball_rect, ball, paddle_rect):
        output = nw.calculateY_NN(ball_rect.centerx - paddle_rect.centerx, ball_rect.centery - paddle_rect.centery, ball_rect.centerx + paddle_rect.centerx, ball_rect.centery + paddle_rect.centery, continuousLearning = True)
        print(output)
        if self.screen_response_area_rect.colliderect(ball_rect):
            if output < 300:
                if not ball.moving_away_from_AI:
                    self.move_up = True
            elif output > 300:
                if not ball.moving_away_from_AI:
                    self.move_down = True
                        
    def reset(self):
        '''reset upon each iteration of update'''
        self.move_up = False
        self.move_down = False
