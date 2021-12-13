#import pandas as pd
import json
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np

def calc_accuracy(y_pred, y_true):
    return accuracy_score(y_true, y_pred)

def calc_F1(y_pred, y_true):
    return f1_score(y_true, y_pred, average='macro')

def calc_means(user_predictions, reference):
    accuracies = []
    f1s = []
    for subj in reference:
        accuracies.append(calc_accuracy(user_predictions[subj], reference[subj]))
        f1s.append(calc_F1(user_predictions[subj], reference[subj]))
    return {"Accuracy": np.mean(accuracies), "F1-score": np.mean(f1s)}

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    #df = pd.read_json(user_submission_file)
    print(test_annotation_file)
    with open(test_annotation_file, "r") as f: 
        reference = json.load(f)
    with open(user_submission_file, "r") as file: 
        user_predictions = json.load(file)
    results = calc_means(user_predictions, reference)
    output = {}

    print("Evaluating for Test Phase")
    output["result"] = [
    {
        "test_split": {
            "Accuracy": results["Accuracy"],
            "F1-score": results["F1-score"]
        }
    },
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]
    print("Completed evaluation for Test Phase")
    print(output)
    return output