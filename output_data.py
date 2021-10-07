import pandas as pd
from sklearn.metrics import accuracy_score

def addNewDataToCsv(nameAlg,runtime,predValues,trueValues,fileName):
    DATA_SHEET = pd.read_csv(fileName + '.csv')

    # Combine the data sheets
    pred_and_truth = pd.DataFrame()
    pred_and_truth['ground_truth'] = trueValues
    pred_and_truth['predictions'] = predValues
    print(pred_and_truth)

    acc = list()
    # values = set(trueValues)
    # print(values)

    for i in range(6):
        DF = pred_and_truth.loc[pred_and_truth['ground_truth'] == i]

        # Calculate the accuracy
        acc.append(accuracy_score(DF['ground_truth'], DF['predictions']))
        print('ACC ' + str(i) + ': ' + str(acc[i]))

    acc_all = accuracy_score(pred_and_truth['ground_truth'], pred_and_truth['predictions'])

    # print(pV)

    DF = pd.DataFrame({'Classifier': nameAlg,
                       'Runtime': runtime,
                       '0': [acc[0]],
                       '1': [acc[1]],
                       '2': [acc[2]],
                       '3': [acc[3]],
                       '4': [acc[4]],
                       '5': [acc[5]],
                       'all': [acc_all]})

    DATA_SHEET = DATA_SHEET.append(DF)
    # print(DATA_SHEET)
    DATA_SHEET.to_csv(fileName + '.csv', index=False)
