

'''
this will contain the actual neural network,
this will teach itself how to drive ...
'''

# LEARN HOW TO MAKE A NEURAL NETWORK AND TRAIN IT

class Tesla(object):
    """docstring for Tesla."""

    def __init__(self):
        super(Tesla, self).__init__()
        # self.arg = arg
        self.posx = 0
        self.posy = 0
        # kind of like x, y coordinates
        # y  = 0 STOP -2 -1 reverse, 1 2 forwards,
        # x  = 0 STRAIGHT, -2 -1 left, 1 2 right
        self.current_direction = (1, 0)

    # return as a string
    def __str__(self):
        return 'Dir='+str(self.current_direction)
    # return as a dictionary
    def __repr__(self):
        pass

    def get_pos(self):
        return (self.posx, self.posy)

    def get_direction(self):
        return self.current_direction

    def set_pos(self, new_x, new_y):
        self.posx = new_x
        self.posy = new_y

    def set_direction(self, newx, newy):
        self.current_direction = newx, newy

    # CAN I MAKE A FUNCTION THAT CAN DETECT TURNS?

    def __nn__(self):
        pass
        # for future alan, the one who knows how to make a nn and train it
