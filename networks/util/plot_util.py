from matplotlib import pyplot as plt
from scipy.misc import toimage

class CifarPlot:
    def __init__(self, x):
        self.x = x
        self.number_images = 9
        self.current_image_idx = 0
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.show_next)
        self.fig.canvas.mpl_connect('button_press_event', self.show_next)
    def show(self):
        self.show_next(None)
        plt.show()
    def show_next(self, event):
        for i in range(0, self.number_images):
            plt.subplot(330 + 1 + i)
            plt.imshow(toimage(self.x[self.current_image_idx + i]))
        self.fig.canvas.draw()
        self.current_image_idx += self.number_images
