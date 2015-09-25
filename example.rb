require_relative 'anomaly_detection'

data = [
  [85, 68, 10.4, 0],
  [88, 62, 12.1, 0],
  [86, 64, 13.6, 0],
  [88, 62, 12.1, 0],
  [86, 64, 13.6, 0],
  [88, 90, 11.1, 1],
  [85, 42, 12.3, 1]
]

ad = AnomalyDetection.new(data)
p ad.eps
p ad.best_f1
p ad.anomaly?([85, 68, 10.4])
p ad.anomaly?([86, 64, 13.6])
p ad.anomaly?([85, 42, 12.3])
p ad.anomaly?([0, 0, 0])
