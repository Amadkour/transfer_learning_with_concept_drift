# evaluate a weighted average ensemble for classification
import time
import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from river import drift
from strlearn.metrics import geometric_mean_score_1, precision, recall, balanced_accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
import numpy as np
# get a list of base models
import helper

K_MAX = 10
num_iterations = 1
metrics = [balanced_accuracy_score, geometric_mean_score_1, f1_score, precision, recall, ]
flows = ['old weights', 'enhance weights-1', 'Concept drift detector']
statistics = [{'time': 0, 'number_of_updates': 0} for _ in flows]
# evaluate each base model
def evaluate_models(X_train, y_train, models):
    # r = RandomForestClassifier(n_estimators=100)

    if len(models) > K_MAX - 1:
        ac = [np.min(c) for c in weights]
        worst_index = ac.index(np.min(ac))
        models.pop(worst_index)
        weights.pop(worst_index)
    g = GradientBoostingClassifier(n_estimators=100)
    g.fit(X_train, y_train)
    models.append((np.unique(y_train), g))


def calc_weights(x, y, models):
    global weights
    classes = np.unique(y)
    weights = [[1.0 for _ in range(len(classes))] for _ in range(len(models))]

    for classifier_index, (_, k) in enumerate(models):
        for class_index, c in enumerate(classes):
            predictions = k.predict(x)
            weight = len(predictions[predictions == c]) / len(y[y == c])
            weights[classifier_index][class_index] = weight
    return weights


def calc_proposed_weights(x, y, models, flow):
    global weights
    classes = np.unique(y)
    weights = [[1.0 for _ in range(len(classes))] for _ in range(len(models))]
    for classifier_index, (_, k) in enumerate(models):
        for class_index, c in enumerate(classes):
            predictions = k.predict(x)
            Wi = (len(predictions[predictions == c]) / len(y[y == c])) * (
                    len(predictions[predictions != c]) / len(y[y != c]))
            weights[classifier_index][class_index] = Wi
    return weights


def ensemble_support_matrix(X, pool_classifiers):
    """Ensemble support matrix."""
    return np.array([member_clf.predict_proba([X]) for member_clf in pool_classifiers])


def predict(data, pool_target_classifiers, pool_source_classifiers, pool_target_weights, pool_source_weights):
    target_clf = [model for y, model in pool_target_classifiers if len(y) == len(pool_target_classifiers[-1][0])]
    source_clf = [model for y, model in pool_source_classifiers if len(y) == len(pool_target_classifiers[-1][0])]
    if len(source_clf) == 0:
        esm_target = ensemble_support_matrix(data, target_clf)
        esm_target = np.array(esm_target).reshape(len(esm_target), len(pool_target_weights[0]))
        Ft = [[0.0 for _ in range(len(pool_target_weights[0]))] for _ in range(len(pool_target_weights))]
        for i in range(len(esm_target)):
            for j in range(len(pool_target_weights[i])):
                Ft[i][j] = esm_target[i][j] * pool_target_weights[i][j]
        target_matrix = np.mean(Ft, axis=0)
        return np.argmax(target_matrix, axis=0)

    else:
        esm_target = ensemble_support_matrix(data, target_clf)
        esm_source = ensemble_support_matrix(data, source_clf)
        esm_target = np.array(esm_target).reshape(len(esm_target), len(pool_target_weights[0]))
        esm_source = np.array(esm_source).reshape(len(esm_source), len(esm_source[0][0]))
        Ft = [[0.0 for _ in range(len(esm_target[0]))] for _ in range(len(esm_target))]

        '''target prediction'''
        '''use esm length instead of weight length because of using subset of classifiers equal to esm length'''
        for i in range(len(esm_target)):
            for j in range(len(esm_target[i])):
                Ft[i][j] = esm_target[i][j] * pool_target_weights[i][j]
        target_matrix = np.argmax(Ft, axis=1)

        # pred = np.argmax(target_matrix, axis=0)
        '''source prediction'''
        F_est = [[(x * pool_source_weights[i]) for j, x in enumerate(esm_source[i])] for i in range(len(esm_source))]
        F_est = np.mean(F_est, axis=0)
        Ft = np.mean(Ft, axis=0)
        F_est = [F_est[i] + Ft[i] for i in range(len(Ft))]
        return np.argmax(F_est, axis=0)


def aw_colar(DS, wS, DT):

    # Unpack source domain
    from scipy import linalg

    XS, yS = DS
    print(XS.shape)

    new_DT = np.zeros(XS.shape)
    new_DT [:np.shape(DT[0])[0],:np.shape(DT[0])[1]]=DT[0]
    # Calculate covariance matrices
    CS = np.cov(XS, rowvar=False) + np.eye(XS.shape[1])
    CT = np.cov(new_DT, rowvar=False) + np.eye(new_DT.shape[1])

    # Apply transformations
    multiplier = linalg.fractional_matrix_power(CS, -0.5)
    multiplier[np.isnan(multiplier)] = 0
    multiplier[np.isinf(multiplier)] = 0
    XS = wS * XS.dot(multiplier)
    multiplier = linalg.fractional_matrix_power(CT, 0.5)
    multiplier[np.isnan(multiplier)] = 0
    multiplier[np.isinf(multiplier)] = 0
    XS = XS.dot(multiplier)

    # Pack the transformed source domain
    DS_p = (np.real(XS), yS)
    return DS_p

    # DS_p now contains the transformed source domain


def projected_data(chunk, sources, source_instance_weights):
    projected_sources = []
    for index, source in enumerate(sources):
        projected_source = aw_colar(source, source_instance_weights[index], chunk)
        projected_sources.append(projected_source)
    return projected_sources


def calc_source_classifiers(data_sources, Tx, Ty, models):
    classifiers = []
    for ite in range(num_iterations):
        for source_index, (Sx, Sy) in enumerate(projected_data((Tx, Ty), data_sources, source_weights)):
            classifier = GradientBoostingClassifier(n_estimators=100)
            classifier.fit(Sx, Sy)
            yhat = classifier.predict(Tx)
            acc = accuracy_score(Ty, yhat)
            source_weights[source_index] = acc
            classifiers.append((np.unique(Sy), classifier))

        for index, (x, y) in enumerate(data_sources):
            yhat = [predict(x[i], models, classifiers, weights, source_weights) for i in range(len(x))]

            accuracy = accuracy_score(y, yhat)
            # Calculate beta for weight multiplication
            beta = 0.5 * np.log(1 + np.sqrt(2 * np.log(x.shape[0] / num_iterations)))
            source_weights[index] = (accuracy * np.exp(-beta * len(y[y == yhat])))
    return classifiers


def calculate_scores(flow_index, chunk_index, y, y_pred):
    scores[flow_index, chunk_index] = [metric(np.array(y), np.array(y_pred)) for metric in metrics]


concept_drift_method = drift.ADWIN()
# fit and evaluate each model
streams = helper.realstreams()
scores = np.zeros((len(flows), streams[list(streams.keys())[0]].n_chunks - 1, len(metrics)))
sources_domain = [streams[list(streams.keys())[0]].get_chunk(), streams[list(streams.keys())[0]].get_chunk(), streams[list(streams.keys())[0]].get_chunk(),
                  streams[list(streams.keys())[0]].get_chunk(), streams[list(streams.keys())[0]].get_chunk(), streams[list(streams.keys())[0]].get_chunk()]
source_weights = [1 for _ in range(len(sources_domain))]


def stream_flow(flow_index,flow):
    global statistics
    streams = helper.realstreams2()
    source_classifiers = []
    weights = []
    models = []
    index = 0
    start_time = time.perf_counter()
    while True:
        print('====[ chunk-', index , ' ]====')
        if streams[list(streams.keys())[0]].is_dry():
            break
        chunk = streams[list(streams.keys())[0]].get_chunk()
        X, y = chunk
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
        if index == 0:
            evaluate_models(X_train, y_train, models)
            if flow == flows[0]:
                weights = calc_weights(X_train, y_train, models)
            else:
                weights = calc_proposed_weights(X_train, y_train, models, flow)
        else:
            times = 0
            if flow == flows[2]:
                concept_drift_method = drift.ADWIN()
                for i in y:
                    concept_drift_method.update(i)
                    if concept_drift_method.drift_detected:
                        times += 1
            else:
                '''to update each chunk if don't concept detector'''
                times = 5
                '''update'''
            if times > 4:
                evaluate_models(X_train, y_train,models)
                if flow == flows[0]:
                    weights = calc_weights(X_train, y_train, models)
                else:
                    weights = calc_proposed_weights(X_train, y_train, models, flow)
                source_classifiers = calc_source_classifiers(sources_domain, X, y,models)
                old_statistics = statistics[flow_index]
                statistics[flow_index]['number_of_updates'] = old_statistics['number_of_updates'] + 1

            yhat = [predict(X_test[i], models, source_classifiers, weights, source_weights) for i in
                    range(len(X_test))]
            calculate_scores(flow_index, index - 1, y_test, yhat)
            # print('========================[ ', flow, ':', scores[flow_index, index - 1], ']')

        index += 1
    finish_time = time.perf_counter()
    statistics[flow_index]['time'] = (finish_time - start_time)
    # Specify the file path where you want to save the data


if __name__ == '__main__':
    start_time = time.perf_counter()
    for flow_index, flow in enumerate(flows):
        stream_flow(flow_index, flow)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")

    np.save(f"output/{list(streams.keys())[0]}/score", scores)
    file_path = f"output/{list(streams.keys())[0]}/statistics.text"
    # Save the list of dictionaries to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(statistics, json_file)

    # from joblib import Parallel, delayed
    #
    # start_time = time.perf_counter()
    # result = Parallel(n_jobs=4, prefer="threads")(
    #     delayed(stream_flow)(flow_index,flow) for flow_index, flow in enumerate(flows))
    # finish_time = time.perf_counter()
    # print(f"Program finished in {finish_time - start_time} seconds")
    # print(result)
