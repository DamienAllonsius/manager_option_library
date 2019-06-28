class ShowRender(object):

    def __init__(self, env):
        self.env = env
        self.display_learning = True
        self.blurred_render = False
        self.gray_scale_render = False
        self.agent_view = False
        self.env.render(blurred_render=self.blurred_render,
                        gray_scale_render=self.gray_scale_render,
                        agent_render=self.agent_view)
        self.env.unwrapped.viewer.window.on_key_press = self.key_press
        self.env.unwrapped.viewer.window.on_key_release = self.key_release

    def display(self):
        if self.display_learning:
            self.env.render(blurred_render=self.blurred_render,
                            gray_scale_render=self.gray_scale_render,
                            agent_render=self.agent_view)

        else:
            self.env.unwrapped.viewer.window.dispatch_events()

    def key_press(self, key, mod):
        if key == ord("d"):
            self.display_learning = not self.display_learning

        if key == ord("b"):
            self.blurred_render = not self.blurred_render

        if key == ord("g"):
            self.gray_scale_render = not self.gray_scale_render

        if key == ord("a"):
            self.agent_view = not self.agent_view

        if key == ord(" "):
            self.agent_view = not self.agent_view
            self.blurred_render = True
            self.gray_scale_render = True

    def key_release(self, key, mod):
        pass


class ShowRenderSwitch(object):
    """
    Like class ShowRender but can only switch from display to not display
    """

    def __init__(self, env):
        self.env = env
        self.display_learning = True
        self.env.render()
        self.env.unwrapped.viewer.window.on_key_press = self.key_press
        self.env.unwrapped.viewer.window.on_key_release = self.key_release

    def display(self):
        if self.display_learning:
            self.env.render()
        else:
            self.env.unwrapped.viewer.window.dispatch_events()

    def key_press(self, key, mod):
        if key == ord("d"):
            self.display_learning = not self.display_learning

    def key_release(self, key, mod):
        pass
