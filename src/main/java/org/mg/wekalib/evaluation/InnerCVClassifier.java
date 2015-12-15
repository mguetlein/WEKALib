package org.mg.wekalib.evaluation;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.mg.javalib.util.StringUtil;
import org.mg.wekalib.evaluation.alt.ClassificationMeasure;
import org.mg.wekalib.evaluation.alt.Measure;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class InnerCVClassifier extends AbstractClassifier
{
	List<Classifier> classifiers = new ArrayList<>();

	List<String> classifierNames = new ArrayList<>();

	Classifier selectedClassifier;

	String selectedClassifierName;

	int numCVFolds = 5;

	long randomSeed = 1;

	Measure measure = ClassificationMeasure.AUC;

	boolean verbose = true;

	public void setClassifiers(List<Classifier> classifiers, List<String> classifierNames)
	{
		this.classifiers = classifiers;
		this.classifierNames = classifierNames;
	}

	public void setMeasure(Measure measure)
	{
		this.measure = measure;
	}

	public Measure getMeasure()
	{
		return measure;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception
	{
		double best = 0;
		int idx = 0;
		for (Classifier classi : classifiers)
		{
			if (verbose)
				System.out.println("start inner-cv of " + classifierNames.get(idx));
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(classi, data, numCVFolds, new Random(randomSeed), new Object[0]);
			double res = measure.getValue(eval, 0);
			if (verbose)
				System.out.println(measure + " : " + StringUtil.formatDouble(res));
			if (selectedClassifier == null || (measure.lowerIsBetter() ? res < best : res > best))
			{
				best = res;
				selectedClassifier = classi;
				selectedClassifierName = classifierNames.get(idx);
			}
			idx++;
		}
		if (verbose)
			System.out.println("Selected " + selectedClassifierName + " (" + measure + " "
					+ StringUtil.formatDouble(best) + ")\n");
		selectedClassifier.buildClassifier(data);
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception
	{
		return selectedClassifier.classifyInstance(instance);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception
	{
		return selectedClassifier.distributionForInstance(instance);
	}

	public static void demo() throws Exception
	{
		//String f = "/home/martin/workspace/DreamChallenge/cache/Top10Sy.QuarterWt.Cellline_Drug_Gex.train.arff";
		//		String f = "/home/martin/workspace/external/weka-3-7-12/data/iris.arff";
		String f = "/home/martin/workspace/external/weka-3-7-12/data/breast-cancer.arff";
		//String f = "/home/martin/workspace/external/weka-3-7-12/data/cpu.arff";
		Instances inst = new Instances(new FileReader(f));
		inst.setClassIndex(inst.numAttributes() - 1);
		//		RemovePercentage rm = new RemovePercentage();
		//		rm.setPercentage(66);
		//		rm.setInputFormat(inst);
		//		inst = Filter.useFilter(inst, rm);
		System.out.println(inst.numInstances() + "\n");

		InnerCVClassifier cv = new InnerCVClassifier();
		cv.setMeasure(ClassificationMeasure.AUC);
		List<Classifier> classifiers = new ArrayList<>();
		List<String> names = new ArrayList<>();
		for (Double c : new Double[] { 0.01, 0.1, 1.0, 10.0 })
		{
			for (Double g : new Double[] { 0.01, 0.1, 1.0, 10.0 })
			{
				for (boolean rbf : new boolean[] { true, false })
				{
					SMO smo = new SMO();
					smo.setC(c);
					String name = "SMO c" + c;
					if (!rbf && g != 1.0)
						continue;
					smo.setKernel(rbf ? new RBFKernel() : new PolyKernel());
					if (rbf)
					{
						((RBFKernel) smo.getKernel()).setGamma(g);
						name += " rbf g" + g;
					}
					else
						name += " poly";
					classifiers.add(smo);
					names.add(name);
				}
			}
		}
		classifiers.add(new RandomForest());
		names.add("Random Forest");
		//		classifiers.add(new LinearRegression());
		//		names.add("LinearRegression");
		//		classifiers.add(new M5P());
		//		names.add("M5P");
		//		classifiers.add(new AlternatingModelTree());
		//		names.add("AlternatingModelTree");
		cv.setClassifiers(classifiers, names);

		Evaluation eval = new Evaluation(inst);
		eval.crossValidateModel(cv, inst, 5, new Random(1), new Object[0]);

		System.out.println("Final cross-validated performance: "
				+ StringUtil.formatDouble(cv.getMeasure().getValue(eval, 1)) + "\n");

		System.out.println("build on entire data");
		cv.buildClassifier(inst);
	}

	public static void main(String[] args) throws Exception
	{
		demo();
	}
}
