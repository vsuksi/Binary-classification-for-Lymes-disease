import calcom
import make_artificial_data as mad


n = 1000
data,labels = mad.make_artificial_data(n)

ce = calcom.classifiers.CentroidencoderClassifier()
nn = calcom.classifiers.NeuralnetworkClassifier()


#ce.fit(data[:4*n//5], labels[:4*n//5])
#pred_ce = ce.predict(data[4*n//5:])

nn.fit(data[:4*n//5], labels[:4*n//5])
pred_nn = nn.predict(data[4*n//5:])

cm = calcom.metrics.ConfusionMatrix()

#print('Centroid:')
#result = cm.evaluate(labels[4*n//5:],pred_ce)
#print(result)

print('Neural Net:')
result = cm.evaluate(labels[4*n//5:],pred_nn)
print(result)

