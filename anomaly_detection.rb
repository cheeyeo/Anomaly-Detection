require "narray"
require "pry"
require "securerandom"

class AnomalyDetection
  SQRT2PI = Math.sqrt(2 * Math::PI)
  
  attr_reader :eps, :best_f1

  def initialize(examples, opts={})
    @best_f1 = 0.0
    @eps = 0.0
    train(examples, opts)
  end

  def train(examples, opts)
    anomalies, non_anomalies = examples.partition{|i| i[-1] > 0}
    training_examples, test_examples = partition(non_anomalies)
    test_examples.concat(anomalies)

    training_examples = training_examples.map { |e| e[0..-2] }
    @m = training_examples.size
    @n = training_examples.first.size

    training_examples = NMatrix.to_na(training_examples)
    @mean = training_examples.mean(1).to_a
    @std = training_examples.stddev(1).to_a
    @std.map! { |std| (std == 0 || std.nan?) ? Float::MIN : std }

    # fit p(x) on training_examples
    # TODO: Vectorize below operation
    pval = []
    training_examples.to_a.each do |x|
      pval << probability(x)
    end

    @eps, @best_f1 = select_threshold(test_examples, pval)
  end

  def select_threshold(examples, pval)
    best_epsilon = 0.0
    best_f1 = 0.0
    f1 = 0.0
    max = pval.max.to_f
    min = pval.min.to_f
    stepsize = (max-min)/1000.0
    stepsize = 0.001 if stepsize.zero?
    min.step(max, stepsize).each do |epss|
      f1 = compute_f1_score(examples, epss)
      if f1 > best_f1
        best_f1 = f1
        best_epsilon = epss
      end
    end

    [best_epsilon, best_f1]
  end

  def compute_f1_score(examples, eps)
    tp = 0
    fp = 0
    fn = 0
    examples.each do |example|
      act = example.last != 0
      pred = self.anomaly?(example[0..-2], eps)
      if (pred && act)
        tp += 1
      elsif (pred && !act)
        fp += 1
      elsif (!pred && act)
        fn += 1
      end
    end
    f1_score(tp, fp, fn)
  end

  def f1_score(tp, fp, fn)
    precision = tp / (tp + fp).to_f
    recall = tp / (tp + fn).to_f
    score = 2.0 * precision * recall / (precision + recall)
    score.nan? ? 0.0 : score
  end

  def probability(x)
    @n.times.map do |i|
      p = normal_pdf(x[i], @mean[i], @std[i])
      (p.nan? || p > 1) ? 1 : p
    end.reduce(1, :*)
  end

  def anomaly?(x, eps = @eps)
    probability(x) < eps
  end

  def normal_pdf(x, mean = 0, std = 1)
    1 / (SQRT2PI * std) * Math.exp(-((x - mean)**2 / (2.0 * (std**2))))
  end

  def partition(examples, percent=0.2)
    examples.shuffle!(random: SecureRandom.random_number(1000_000))
    n = (examples.size * percent).floor
    [examples[n..-1], examples[0...n]]
  end
end
