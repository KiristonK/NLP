

from data import DataWorker

# dla trenowania modelu
#dw = DataWorker('complaints_processed.csv')
#dw.create_predictor()

dw = DataWorker()
dw.load_predictor()


dw.save_predictor()

print('\n\nTest input: shit payment')
pr = dw.predict_string('shit payment')
print('Text classified as: ' + pr[0])

print('Test input: nice place')
pr = dw.predict_string('nice place')
print('Text classified as: ' + pr[0])

print('Test input: hate employees, they are unpleasant')
pr = dw.predict_string('hate employees, they are unpleasant')
print('Text classified as: ' + pr[0])

print('Test input: within day closing told using lender would cause purchase agreement price go even appraisal done also give option purchasing loan without pmi said absolutely possible loan le include pmi forcing choose product want signed contract sale consultant told use lender losing incentive closing cost')
pr = dw.predict_string('within day closing told using lender would cause purchase agreement price go even appraisal done also give option purchasing loan without pmi said absolutely possible loan le include pmi forcing choose product want signed contract sale consultant told use lender losing incentive closing cost')
print('Text classified as: ' + pr[0])

print('Test input: creditor refuse provide income statement account receivable account payable')
pr = dw.predict_string('creditor refuse provide income statement account receivable account payable')
print('Text classified as: ' + pr[0])
