import cv2
from gym.envs.classic_control import rendering


class ShowRender(object):

    def __init__(self):
        self.display_learning = True

        self.vanilla_view = True
        self.option_view = False
        self.agent_view = False

        self.viewer = rendering.SimpleImageViewer()
        
    def render(self, observation):
        """
        :param observation: a dictionary containing the observations:
        - obersation vanilla
        - observation agent
        - observation option
        :return:
        """
        if self.display_learning:
            if type(observation).__name__ == "ndarray":
                self.display(observation)

            else:
                assert list(observation.keys()) == ["vanilla", "agent", "option"], \
                    "observation must be a dictionary with 3 keys : vanilla, agent and option"

                if self.vanilla_view:
                    self.display(observation["vanilla"])

                elif self.agent_view:
                    self.display(observation["agent"])

                elif self.option_view:
                    self.display(observation["option"])

    def display(self, image_pixel):
        img = cv2.resize(image_pixel, (512, 512), interpolation=cv2.INTER_NEAREST)
        self.viewer.imshow(img)

        if self.viewer.window is not None:
            self.viewer.window.on_key_press = self.key_press

    def close(self):
        self.viewer.close()

    def key_press(self, key, mod):
        if key == ord("d"):
            print("press d to display the observation")
            self.display_learning = not self.display_learning

        if key == ord("o"):
            self.set_option_view()

        if key == ord("v"):
            self.set_vanilla_view()

        if key == ord("a"):
            self.set_agent_view()

        if key == ord(" "):

            if self.option_view:
                self.set_vanilla_view()

            elif self.vanilla_view:
                self.set_agent_view()

            elif self.agent_view:
                self.set_option_view()

    def set_vanilla_view(self):
        print("original view")
        self.vanilla_view = True
        self.option_view = False
        self.agent_view = False

    def set_option_view(self):
        print("option's view")
        self.vanilla_view = False
        self.option_view = True
        self.agent_view = False

    def set_agent_view(self):
        print("agent's view")
        self.vanilla_view = False
        self.option_view = False
        self.agent_view = True
