import argparse
import cvbaselineutils.yolo as yolo
import cvbaselineutils.config as conf
import cvdatasetutils.vgopd as vgopd

def generate_description():
    return """
        Command line tool interface with the Yolo module.
    """


def defineParser():
    parser = argparse.ArgumentParser(description=generate_description())

    parser.add_argument('action', type=str, help='Action to execute. Execute "list" for a list of available actions')
    parser.add_argument('--modelFile', type=str, help='Name of a model to load for evaluation or finetunning')

    return parser


def show_actions_list():
    actions_list = """
        - train:            train a module from scratch with the default configuration
        - evaluate:         evaluate the selected model file
    """

    print(actions_list)


def execute(action, modelFile):
    if action == "train":
        yolo.execute(modelFile)
    elif action == "evaluate":
        yolo.evaluate(modelFile)
    else:
        print("Unrecognized action\n")


def start():
    conf.CUDA_DEVICE="0"

    args = defineParser().parse_args()
    execute(args.action, args.modelFile)


if __name__== "__main__":
    start()

