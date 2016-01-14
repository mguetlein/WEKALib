package org.mg.wekalib.evaluation;

import java.beans.IntrospectionException;
import java.beans.PropertyDescriptor;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.swing.DefaultListModel;

import org.apache.commons.lang3.SerializationUtils;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.FileUtil;
import org.mg.javalib.util.ListUtil;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Utils;
import weka.experiment.CrossValidationResultProducer;
import weka.experiment.Experiment;
import weka.experiment.InstancesResultListener;
import weka.experiment.OutputZipper;
import weka.experiment.PropertyNode;
import weka.experiment.SplitEvaluator;

public class Crossvalidate
{
	public static final String CACHE_DIR = System.getProperty("user.home") + "/.weka/cache/";

	//	static ParallelHandler jobs = new ParallelHandler(2);

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static void run(String arffFile, final boolean classification, final Classifier classifier, int numFolds,
			int run, final String outfile, final String predictionsFile)//, boolean cache) String classifierParamAsString, 
			throws Exception
	{
		//		final String cacheOutFile = CACHE_DIR + "cv-" + numFolds + "-" + run + "-"
		//				+ classifier.getClass().getSimpleName() + classifierParamAsString + "-"
		//				+ StringUtil.getMD5(arffFile + "###" + FileUtil.readStringFromFile(arffFile));

		//		if (cache && new File(cacheOutFile).exists() && new File(cacheOutFile).length() > 0)
		//		{
		//			// do nothing
		//		}
		//		else

		File tmp = File.createTempFile("weka", "exp");
		System.out.println("tmp " + tmp);

		//		int maxBuildCount = numFolds * classi.length;
		Experiment exp = new Experiment();
		exp.setPropertyArray(new Classifier[0]);
		exp.setUsePropertyIterator(true);

		SplitEvaluator se = null;
		Classifier sec = null;

		if (classification)
		{
			se = new MyClassifierSplitEvaluator();
			sec = ((MyClassifierSplitEvaluator) se).getClassifier();
		}
		else
		{
			se = new MyRegressionSplitEvaluator();
			sec = ((MyRegressionSplitEvaluator) se).getClassifier();
		}

		final StringBuffer done = new StringBuffer("");

		class MyCrossValidationResultProducer extends CrossValidationResultProducer
		{
			public Predictions predictions;

			//				@Override
			//				public void setSplitEvaluator(SplitEvaluator newSplitEvaluator)
			//				{
			//					System.err.println("setting evaluator");
			//					MyRegressionSplitEvaluator ev = new MyRegressionSplitEvaluator();
			//					ev.setClassifier(classifier);
			//					super.setSplitEvaluator(ev);
			//				}

			//				public void spearman()
			//				{
			//					SpearmansCorrelation sp = new SpearmansCorrelation();
			//					ArrayList<Prediction> p = m_delegate.predictions();
			//					double actual[] = new double[p.size()];
			//					double predicted[] = new double[p.size()];
			//					int i = 0;
			//					for (Prediction pred : p)
			//					{
			//						actual[i] = pred.actual();
			//						predicted[i] = pred.predicted();
			//						i++;
			//					}
			//					double spearman = sp.correlation(actual, predicted);
			//				}

			public void doRun(int run) throws Exception
			{
				//					if (!(m_SplitEvaluator instanceof MyRegressionSplitEvaluator))
				//					{
				//						MyRegressionSplitEvaluator ev = new MyRegressionSplitEvaluator();
				//						ev.setClassifier(classifier);
				//						setSplitEvaluator(ev);
				//					}

				if (getRawOutput())
				{
					if (m_ZipDest == null)
					{
						m_ZipDest = new OutputZipper(m_OutputFile);
					}
				}

				if (m_Instances == null)
				{
					throw new Exception("No Instances set");
				}

				final PairwiseCVSplitter cv = new PairwiseCVSplitter(m_Instances, m_NumFolds, new Random(run));

				//					// Randomize on a copy of the original dataset
				//					Instances runInstances = new Instances(m_Instances);
				//
				//					Random random = new Random(run);
				//					runInstances.randomize(random);
				//					if (runInstances.classAttribute().isNominal())
				//					{
				//						runInstances.stratify(m_NumFolds);
				//					}

				for (int fold = 0; fold < m_NumFolds; fold++)
				{
					System.out.println("fold " + (fold + 1) + "/" + m_NumFolds);

					// Add in some fields to the key like run and fold number, dataset name
					Object[] seKey = m_SplitEvaluator.getKey();
					Object[] key = new Object[seKey.length + 3];
					key[0] = Utils.backQuoteChars(m_Instances.relationName());
					key[1] = "" + run;
					key[2] = "" + (fold + 1);
					System.arraycopy(seKey, 0, key, 3, seKey.length);

					if (m_ResultListener.isResultRequired(this, key))
					{
						//							Instances train = runInstances.trainCV(m_NumFolds, fold, random);
						//							Instances test = runInstances.testCV(m_NumFolds, fold);

						Instances train = cv.getTrain(fold);
						Instances test = cv.getTest(fold);

						try
						{
							Object[] seResults = m_SplitEvaluator.getResult(train, test);
							//Object[] seResults = cvResults.get(fold);

							double act[] = new double[test.numInstances()];
							for (int i = 0; i < test.numInstances(); i++)
								act[i] = test.get(i).value(test.classAttribute());
							double prd[];
							double conf[];
							if (classification)
							{
								prd = ((MyClassifierSplitEvaluator) m_SplitEvaluator).getPredictions();
								conf = ((MyClassifierSplitEvaluator) m_SplitEvaluator).getProbabilty();
							}
							else
							{
								prd = ((MyRegressionSplitEvaluator) m_SplitEvaluator).getPredictions();
								conf = ((MyRegressionSplitEvaluator) m_SplitEvaluator).getLogDensity();
							}
							predictions = PredictionUtil
									.add(predictions, fold, act, prd, conf, cv.getTestIndices(fold));

							Object[] results = new Object[seResults.length + 1];
							System.out.println("num results: " + results.length);

							results[0] = getTimestamp();
							System.arraycopy(seResults, 0, results, 1, seResults.length);
							if (m_debugOutput)
							{
								String resultName = ("" + run + "." + (fold + 1) + "."
										+ Utils.backQuoteChars(m_Instances.relationName()) + "." + m_SplitEvaluator
										.toString()).replace(' ', '_');
								resultName = Utils.removeSubstring(resultName, "weka.classifiers.");
								resultName = Utils.removeSubstring(resultName, "weka.filters.");
								resultName = Utils.removeSubstring(resultName, "weka.attributeSelection.");
								m_ZipDest.zipit(m_SplitEvaluator.getRawResultOutput(), resultName);
							}
							m_ResultListener.acceptResult(this, key, results);
						}
						catch (Exception ex)
						{
							// Save the train and test datasets for debugging purposes?
							throw ex;
						}
					}
				}
				done.append("true");
			}
		}
		MyCrossValidationResultProducer cvrp = new MyCrossValidationResultProducer();
		cvrp.setNumFolds(numFolds);
		cvrp.setSplitEvaluator(se);

		PropertyNode[] propertyPath = new PropertyNode[2];
		try
		{
			propertyPath[0] = new PropertyNode(se, new PropertyDescriptor("splitEvaluator",
					CrossValidationResultProducer.class), CrossValidationResultProducer.class);
			propertyPath[1] = new PropertyNode(sec, new PropertyDescriptor("classifier", se.getClass()), se.getClass());
		}
		catch (IntrospectionException e)
		{
			e.printStackTrace();
		}

		exp.setResultProducer(cvrp);
		exp.setPropertyPath(propertyPath);

		exp.setRunLower(run);
		exp.setRunUpper(run);

		exp.setPropertyArray(new Classifier[] { classifier });

		DefaultListModel model = new DefaultListModel();
		model.addElement(new File(arffFile));
		exp.setDatasets(model);

		InstancesResultListener irl = new InstancesResultListener();
		//			File resultsFile;
		//		if (outfile == null)
		//			resultsFile = File.createTempFile("results", "arff");
		//		else
		irl.setOutputFile(tmp);
		exp.setResultListener(irl);

		// 2. run experiment
		//		System.out.println("Initializing...");
		exp.initialize();
		//		System.out.println("Running...");
		exp.runExperiment();
		//		System.out.println("Finishing...");
		exp.postProcess();

		if (done.toString().equals("true"))
		{
			SerializationUtils.serialize(cvrp.predictions, new FileOutputStream(predictionsFile));
			FileUtil.robustRenameTo(tmp.getAbsolutePath(), outfile);// cacheOutFile);
			System.out.println("results printed to " + outfile);// cacheOutFile);
			//		FileUtil.copy(cacheOutFile, outfile);
		}
		else
			throw new IllegalStateException("cross-validation failed");
	}

	static class PairwiseCVSplitter
	{
		Instances inst;
		int numFolds;
		List<Integer[]> cvIdx;

		public PairwiseCVSplitter(Instances inst, int numFolds, Random r)
		{
			this.inst = new Instances(inst);
			this.numFolds = numFolds;

			if (inst.size() % 2 == 1)
				throw new IllegalStateException();

			Integer[] idx = ArrayUtil.toIntegerArray(ArrayUtil.indexArray(inst.size() / 2));
			ArrayUtil.scramble(idx, r);
			cvIdx = ArrayUtil.split(idx, numFolds);
		}

		public Instances getTrain(int fold)
		{
			return get(fold, true);
		}

		public Instances getTest(int fold)
		{
			return get(fold, false);
		}

		public int[] getTestIndices(int fold)
		{
			List<Integer> indices = new ArrayList<>();
			for (Integer i : cvIdx.get(fold))
			{
				indices.add(2 * i);
				indices.add(2 * i + 1);
			}
			return ArrayUtil.toPrimitiveIntArray(indices);
		}

		private Instances get(int fold, boolean train)
		{
			ArrayList<Attribute> attributes = new ArrayList<>();
			for (int i = 0; i < inst.numAttributes(); i++)
				attributes.add(inst.attribute(i));
			Instances data = new Instances((train ? "Train" : "Test") + " fold of " + inst.relationName(), attributes,
					0);
			data.setClassIndex(inst.classIndex());
			for (int f = 0; f < numFolds; f++)
			{
				if ((train && fold != f) || (!train && fold == f))
				{
					for (Integer i : cvIdx.get(f))
					{
						data.add(inst.get(2 * i));
						data.add(inst.get(2 * i + 1));
					}
				}
			}
			return data;
		}

		public static void test()
		{
			Attribute attr = new Attribute("A");
			Instances data = new Instances("Data", (ArrayList<Attribute>) ListUtil.createList(attr), 0);
			for (int i = 0; i < 12; i++)
				for (int j = 0; j < 2; j++)
					data.add(new DenseInstance(1.0, new double[] { i }));
			System.out.println("data:\n" + data);

			int numFolds = 5;
			PairwiseCVSplitter cv = new PairwiseCVSplitter(data, numFolds, new Random(123));
			for (int f = 0; f < numFolds; f++)
			{
				System.out.println("\nfold " + f);
				System.out.println("\ntrain");
				System.out.println(cv.getTrain(f));
				System.out.println("\ntest");
				System.out.println(cv.getTest(f));

				System.out.println("\ntest-indices:");
				System.out.println(ArrayUtil.toString(cv.getTestIndices(f)));
			}
		}
	}

	public static void main(String[] args) throws Exception
	{
		PairwiseCVSplitter.test();

		//		run("/tmp/data1.arff", 10, "/tmp/outfile1");
		//		run("/tmp/data2.arff", 10, "/tmp/outfile2");
		//		ResultsParser.parse("/tmp/outfile1", "/tmp/outfile2");

	}
}
