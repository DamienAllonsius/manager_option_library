import os
import time


class SaveResults(object):

    def __init__(self, parameters):
        self.parameters = parameters
        self.dir_path = self.get_dir_path()
        self.file_results_name = None

    @staticmethod
    def get_dir_path():
        dir_name = "results/"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        dir_name += time.asctime(time.localtime(time.time())).replace(" ", "_")
        os.mkdir(dir_name)
        print("results are stored in directory: " + str(dir_name))

        return dir_name

    def write_message(self, message):
        """
        :param message:
        :return:
        """
        f = open(self.file_results_name, "a")
        f.write(message)
        f.close()

    def write_message_in_a_file(self, file_name, message):
        f = open(self.dir_path + "/" + file_name, "a")
        f.write(message)
        f.close()

    def write_reward(self, t, total_reward):
        f = open(self.file_results_name, "a")
        f.write("t = " + str(t) + " reward = " + str(total_reward) + "\n")
        f.close()

    def write_setting(self):
        """
        writes the parameters in a file
        """
        f = open(self.dir_path + "/" + "setting", "a")
        for key in self.parameters:
            f.write(key + " : " + str(self.parameters[key]) + "\n")
        f.write("\n" * 3)
        f.close()

    def set_file_results_name(self, seed):
        self.file_results_name = self.dir_path + "/" + "seed_" + str(seed)
