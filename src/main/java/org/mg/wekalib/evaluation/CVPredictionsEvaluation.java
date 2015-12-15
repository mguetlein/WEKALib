package org.mg.wekalib.evaluation;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.mg.javalib.util.ArrayUtil;
import org.mg.wekautil.Predictions;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.ConditionalDensityEstimator;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.classifiers.meta.RegressionByDiscretization;
import weka.core.BatchPredictor;
import weka.core.Instances;

public class CVPredictionsEvaluation extends Evaluation
{
	public static class MyDelegate extends weka.classifiers.evaluation.Evaluation
	{
		Predictions cvPredictions = new Predictions();

		private int fold = 0;
		private List<Integer> origTestIndices = new ArrayList<>();

		public MyDelegate(Instances data) throws Exception
		{
			super(data);
		}

		@Override
		public void crossValidateModel(Classifier classifier, Instances data, int numFolds, Random random,
				Object... forPredictionsPrinting) throws Exception
		{
			// Make a copy of the data we can reorder
			data = new Instances(data);

			// --- changes by MG - start ------------------------------------
			cvPredictions = new Predictions();
			fold = 0;
			double origWeights[] = new double[data.size()];
			for (int i = 0; i < data.numInstances(); i++)
			{
				origWeights[i] = data.instance(i).weight();
				data.instance(i).setWeight(i);
			}
			// --- changes by MG - end ------------------------------------

			data.randomize(random);
			if (data.classAttribute().isNominal())
			{
				data.stratify(numFolds);
			}

			// We assume that the first element is a
			// weka.classifiers.evaluation.output.prediction.AbstractOutput object
			AbstractOutput classificationOutput = null;
			if (forPredictionsPrinting.length > 0)
			{
				// print the header first
				classificationOutput = (AbstractOutput) forPredictionsPrinting[0];
				classificationOutput.setHeader(data);
				classificationOutput.printHeader();
			}

			// Do the folds
			for (int i = 0; i < numFolds; i++)
			{
				Instances train = data.trainCV(numFolds, i, random);

				// --- changes by MG - start ------------------------------------
				System.out.println("fold " + i);
				for (int j = 0; j < train.numInstances(); j++)
				{
					int origIdx = (int) train.instance(j).weight();
					train.instance(j).setWeight(origWeights[origIdx]);
				}
				fold = i;
				// --- changes by MG - end ------------------------------------

				setPriors(train);
				Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
				copiedClassifier.buildClassifier(train);
				Instances test = data.testCV(numFolds, i);

				// --- changes by MG - start ------------------------------------
				origTestIndices.clear();
				for (int j = 0; j < test.numInstances(); j++)
				{
					int origIdx = (int) test.instance(j).weight();
					origTestIndices.add(origIdx);
					test.instance(j).setWeight(origWeights[origIdx]);
				}
				// --- changes by MG - end ------------------------------------

				evaluateModel(copiedClassifier, test, forPredictionsPrinting);
			}
			m_NumFolds = numFolds;

			if (classificationOutput != null)
			{
				classificationOutput.printFooter();
			}
		}

		public double[] evaluateModel(Classifier classifier, Instances data, Object... forPredictionsPrinting)
				throws Exception
		{
			// for predictions printing
			AbstractOutput classificationOutput = null;

			double predictions[] = new double[data.numInstances()];

			if (forPredictionsPrinting.length > 0)
			{
				classificationOutput = (AbstractOutput) forPredictionsPrinting[0];
			}

			if (classifier instanceof BatchPredictor
					&& ((BatchPredictor) classifier).implementsMoreEfficientBatchPrediction())
			{
				// make a copy and set the class to missing
				Instances dataPred = new Instances(data);
				for (int i = 0; i < data.numInstances(); i++)
				{
					dataPred.instance(i).setClassMissing();
				}
				double[][] preds = ((BatchPredictor) classifier).distributionsForInstances(dataPred);
				for (int i = 0; i < data.numInstances(); i++)
				{
					double[] p = preds[i];
					predictions[i] = evaluationForSingleInstance(p, data.instance(i), true);
					if (classificationOutput != null)
					{
						classificationOutput.printClassification(p, data.instance(i), i);
					}
				}
			}
			else
			{
				// Need to be able to collect predictions if appropriate (for AUC)

				for (int i = 0; i < data.numInstances(); i++)
				{
					predictions[i] = evaluateModelOnceAndRecordPrediction(classifier, data.instance(i));

					// --- changes by MG - start ------------------------------------
					double conf = Double.NaN;
					if (classifier instanceof ConditionalDensityEstimator)
						conf = ((ConditionalDensityEstimator) classifier).logDensity(data.instance(i), predictions[i]);
					PredictionUtil.add(cvPredictions, data.instance(i).classValue(), predictions[i], conf, fold,
							origTestIndices.isEmpty() ? -1 : origTestIndices.get(i));
					// --- changes by MG - end ------------------------------------

					if (classificationOutput != null)
					{
						classificationOutput.printClassification(classifier, data.instance(i), i);
					}
				}
			}
			return predictions;
		}
	}

	public CVPredictionsEvaluation(Instances data) throws Exception
	{
		super(data);
		m_delegate = new MyDelegate(data);
	}

	public Predictions getCvPredictions()
	{
		return ((MyDelegate) m_delegate).cvPredictions;
	}

	public static void main(String[] args)
	{
		try
		{
			Instances inst = new Instances(new FileReader("/home/martin/data/weka/numeric/baskball.arff"));
			inst.setClassIndex(inst.numAttributes() - 1);
			CVPredictionsEvaluation eval = new CVPredictionsEvaluation(inst);
			eval.crossValidateModel(new RegressionByDiscretization(), inst, 10, new Random(), new Object[0]);
			System.out.println(eval.correlationCoefficient());

			Predictions p = eval.getCvPredictions();
			System.out.println(ArrayUtil.toString(p.actual));
			System.out.println(ArrayUtil.toString(p.predicted));
			System.out.println(ArrayUtil.toString(p.confidence));
			System.out.println(ArrayUtil.toString(p.fold));
			System.out.println(ArrayUtil.toString(p.origIndex));
			System.out.println(PredictionUtil.pearson(p));
		}
		catch (Exception e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
