

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
        # 0 - reverse, 1 - forwards, 2 - left, 3 - right
        self.current_direction = 1

    # return as a string
    def __str__(self):
        return 'Position=('+str(self.posx)+','+str(self.posy)+'), Dir='+str(self.current_direction)
    # return as a dictionary
    def __repr__(self):
        pass

    def __nn__(self):
        pass
        # for future alan, the one who knows how to make a nn and train it

    def get_pos(self):
        return (self.posx, self.posy)

    def get_direction(self):
        return self.current_direction
