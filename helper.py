import numpy as np
from scipy.io.matlab._streams import make_stream
from sklearn.datasets import make_classification
from strlearn.streams import ARFFParser, StreamGenerator


def realstreams():
    data = ARFFParser("datasets/coverType.arff", n_chunks=200, chunk_size=2000)
    return {
        "coverType": data,
    }


def realstreams2():
    # data=ARFFParser("datasets/power.arff", n_chunks=200, chunk_size=2000)
    data = ARFFParser("datasets/sensors.arff", n_chunks=200, chunk_size=2000)
    return {
        "sensors": data,
    }


def steaming2():
    streams = {}
    n_classes = 4
    n_features = 10
    n_chunks = 200
    drift_type = False
    spacing = 5
    stream = StreamGenerator(n_chunks=n_chunks, chunk_size=2000, n_features=n_features, n_drifts=20,
                             n_classes=n_classes,
                             y_flip=n_features * 0.7, concept_sigmoid_spacing=spacing, incremental=drift_type,
                             n_informative=5,

                             )
    if spacing is None and drift_type == True:
        pass
    else:
        streams.update({str(stream): stream})

    return streams


def streaam2():
    import numpy as np
    from sklearn.datasets import make_classification

    # Set the seed for reproducibility
    np.random.seed(42)

    # Generate an initial dataset with two existing classes
    X, y = make_classification(
        n_samples=300000,
        n_features=10,
        n_informative=5,
        scale=1,
        n_classes=4,
        random_state=42
    )

    # Generate an emerging new class dataset
    X_emerging, _ = make_classification(
        n_samples=100000,
        n_features=10,
        n_informative=5,
        scale=100,
        n_classes=1,
        random_state=42
    )
    y_emerging = np.ones(100000) + 4

    # Concatenate the existing dataset with the emerging new class dataset
    X = np.concatenate((X, X_emerging), axis=0)
    y = np.concatenate((y, y_emerging), axis=0)

    # Shuffle the dataset
    shuffle_indices = np.random.permutation(X.shape[0])
    X = X[shuffle_indices]
    y = y[shuffle_indices]
    # X2, y2 = make_classification(
    #     n_samples=150000,
    #     n_features=5,
    #     n_informative=2,
    #     n_clusters_per_class=1,
    #     n_classes=3,
    #     random_state=333
    # )
    # Concatenate the existing dataset with the emerging new class dataset
    X = np.concatenate((X, X_emerging), axis=0)
    y = np.concatenate((y, y_emerging), axis=0)

    # Shuffle the dataset
    shuffle_indices = np.random.permutation(X.shape[0])
    # X = X[shuffle_indices]
    # y = y[shuffle_indices]
    with open('../datasets/synthetic.arff', 'a') as file:
        file.write('''@relation synthetic
@attribute att0 numeric
@attribute att1 numeric
@attribute att2 numeric
@attribute att3 numeric
@attribute att4 numeric
@attribute att5 numeric
@attribute att6 numeric
@attribute att7 numeric
@attribute att8 numeric
@attribute att9 numeric
@attribute class {0,1,2,3,5}
@data
''')
        # Append the content to the file
        for i in range(len(X)):
            s = ''
            for j in range(len(X[i])):
                if (j == 9):
                    s += str(X[i][j]) + ',' + str(y[i])
                else:
                    s += str(X[i][j]) + ','
            file.write(s + '\n')
        file.close()


def streams():
    # Variables
    # distributions = [[0.95, 0.05], [0.90, 0.10], [0.85, 0.15]]
    distributions = [[0.27, 0.23, 0.3, 0.2]]
    label_noises = [
        0.21,
    ]
    incremental = [True]
    ccs = [5]
    n_drifts = 50

    # Prepare streams
    streams = {}
    for drift_type in incremental:
        for distribution in distributions:
            for flip_y in label_noises:
                for spacing in ccs:
                    stream = StreamGenerator(
                        incremental=drift_type,
                        weights=distribution,
                        random_state=123,
                        y_flip=flip_y,
                        concept_sigmoid_spacing=spacing,
                        n_drifts=n_drifts,
                        chunk_size=2000,
                        n_chunks=200,
                        n_clusters_per_class=1,
                        n_classes=4,
                        n_features=8,
                        n_informative=8,
                        n_redundant=0,
                        n_repeated=0,
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        streams.update({str(stream): stream})

    return streams

# streaam2()
