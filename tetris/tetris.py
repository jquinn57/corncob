import sys, pygame
import numpy as np
import time

# sounds from https://www.pacdv.com/sounds/interface_sounds-3.html

class Tetris:
    pieces = np.array( [[1, 7], 
                        [4, 7], 
                        [6, 6], 
                        [0, 15], 
                        [6, 3], 
                        [3, 6], 
                        [2, 7]], dtype=np.uint8)
    colors = np.array( [ [0xe6, 0x19, 0x4b], 
                         [0xf5, 0x82, 0x31], 
                         [0xff, 0xe1, 0x19], 
                         [0x3c, 0xb4, 0x4b],
                         [0x43, 0x63, 0xd8],
                         [0xe6, 0xbe, 0xff],
                         [0x00, 0x80, 0x80]], dtype=np.uint8)

    def __init__(self, blocksize=30):   
        self.Nx = 12
        self.Ny = 24
        self.pad = 5
        self.blocksize = blocksize
        self.width =  self.Nx * blocksize
        self.height = self.Ny * blocksize
        pygame.init()
        self.still_alive = True
        self.screen = pygame.display.set_mode((self.width*2, self.height))
        self.state = np.zeros((self.Ny, self.Nx), dtype=np.int)
        self.current_piece = 0
        self.next_piece_id = 1
        self.current_pos = np.array([0, 6])
        self.piece_mask = np.zeros((4, 4), dtype=np.uint8)
        self.speed0 = 0.1
        self.drop_sound = pygame.mixer.Sound('./sound97.wav')
        self.clear_sound = pygame.mixer.Sound('./sound110.wav')
        self.cleared_lines = 0
        self.score = 0 
        self.font = pygame.font.Font('freesansbold.ttf', 32) 


    @staticmethod
    def gen_piece_mask(new_piece_id):
        piece_mask = np.zeros((4, 4), dtype=np.uint8)
        piece = Tetris.pieces[new_piece_id, :]
        for y in range(2):
            row_shape = piece[y]
            mask = 8
            for x in range(0, 4):
                if (row_shape & mask) > 0 :
                    piece_mask[y, x] = 1
                mask = mask >> 1
        return piece_mask

    def draw_piece(self, piece_mask, piece_id, pos):
        color = Tetris.colors[piece_id, :]
        for i, y in enumerate(range(pos[0], pos[0] + 4)):
            for j, x in enumerate(range(pos[1], pos[1] + 4)):
                if piece_mask[i, j] > 0:
                    pygame.draw.rect(self.screen, 0.9*color, 
                                     (x*self.blocksize,y*self.blocksize, 
                                     self.blocksize, self.blocksize))
                    pygame.draw.rect(self.screen, 0.7*color, 
                                     (x*self.blocksize+self.pad, y*self.blocksize+self.pad, 
                                     self.blocksize-2*self.pad, self.blocksize-2*self.pad))

    def draw_screen(self, state):
        self.screen.fill((0,0,0))
        gray = (64, 64, 64)
        pygame.draw.rect(self.screen, gray, (self.width, 0, self.width, self.height))
        
        score_str = f'Lines: {self.cleared_lines}'
        text = self.font.render(score_str, True, (255, 255, 255),gray) 
        text_rect = text.get_rect()
        text_rect.center = (self.width * 1.5 , 30)
        self.screen.blit(text, text_rect)

        score_str = f'Score: {self.score}'
        text = self.font.render(score_str, True, (255, 255, 255), gray) 
        text_rect = text.get_rect()
        text_rect.center = (self.width * 1.5 , 70)
        self.screen.blit(text, text_rect)

        for x in range(self.Nx):
            for y in range(self.Ny):
                if state[y , x] > 0:
                    color = Tetris.colors[self.state[y , x]-1, :]
                    pygame.draw.rect(self.screen, 0.9*color, 
                                     (x*self.blocksize,y*self.blocksize, 
                                        self.blocksize, self.blocksize))
                    pygame.draw.rect(self.screen, 0.7*color, 
                                     (x*self.blocksize+self.pad,y*self.blocksize+self.pad, 
                                     self.blocksize-2*self.pad, self.blocksize-2*self.pad))
        self.draw_piece(self.piece_mask, self.current_piece, self.current_pos)
        self.draw_piece(self.next_piece_mask, self.next_piece_id, (6, 16))
        pygame.display.flip()

    def check_move(self, dy, dx, rot=0):
        pad = 3
        pm = self.piece_mask.copy()
        if rot:
            pm = np.rot90(pm)

        padded_state = np.pad(self.state, ((pad,pad), (pad,pad)), constant_values=1)
        state_local = padded_state[pad + self.current_pos[0] + dy: pad + self.current_pos[0] + dy + 4,
                                   pad + self.current_pos[1] + dx: pad + self.current_pos[1] + dx + 4]
        return (np.sum(state_local * pm) == 0)


    def next_piece(self):
        self.current_pos = np.array([0, 6])
        self.piece_mask = self.gen_piece_mask(self.next_piece_id)
        self.current_piece = self.next_piece_id 
        self.next_piece_id = np.random.randint(7)
        self.next_piece_mask = self.gen_piece_mask(self.next_piece_id)


    def update_state(self):
        for i, y in enumerate(range(self.current_pos[0], self.current_pos[0] + 4)):
            for j, x in enumerate(range(self.current_pos[1], self.current_pos[1] + 4)):
                if self.piece_mask[i, j] > 0:
                    self.state[y, x] = self.current_piece + 1


    def check_lines(self):
        cleared_lines = np.all(self.state > 0, axis=1)
        num_cleared = np.sum(cleared_lines)
        self.cleared_lines += num_cleared
        self.score += num_cleared*num_cleared
        if num_cleared > 0:
            cleared_state = self.state.copy()
            cleared_state[cleared_lines, :] = 0
            for _ in range(num_cleared):
                self.clear_sound.play()
                self.draw_screen(cleared_state)
                time.sleep(0.1)
                self.draw_screen(self.state)
                time.sleep(0.1)

            new_state = np.zeros_like(self.state)
            row_idx = np.arange(self.Ny)
            new_row_idx = row_idx[~cleared_lines][::-1]
            y = self.Ny -1
            for j in new_row_idx:
                new_state[y, :] = self.state[j, :]
                y = y-1
            self.state = new_state

    def run(self):
        self.next_piece()
        speed = self.speed0
        dy = 0
        dropped = False
        delay0 = 0.02
        delay = delay0
        while self.still_alive:
            time.sleep(delay)
            self.draw_screen(self.state)
            dy += speed
            dx = 0
            rot = 0
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        dx = 1
                    if event.key == pygame.K_LEFT:
                        dx = -1
                    if event.key == pygame.K_UP:
                        rot = 1
                    if event.key == pygame.K_DOWN:
                        speed = 1.0
                    if event.key == pygame.K_SPACE:
                        self.drop_sound.play() 
                        speed = 1.0
                        dropped = True
                        delay = 0.01

            move_ok = self.check_move(0, dx, rot)
            if move_ok:
                self.current_pos[1] += dx
                if rot:
                    self.piece_mask = np.rot90(self.piece_mask)

            if int(dy) > 1:
                dy = 0 
                move_ok = self.check_move(1, 0)
                if move_ok:
                    self.current_pos[0] += 1
                    if not dropped:
                        speed = self.speed0
                else:
                    self.update_state()
                    self.check_lines()
                    self.next_piece()
                    speed = self.speed0
                    dropped = False
                    delay = delay0


if __name__ == '__main__':
    tetris = Tetris()
    tetris.run()
