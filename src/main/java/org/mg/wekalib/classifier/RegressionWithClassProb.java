package org.mg.wekalib.classifier;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.mg.javalib.util.ListUtil;
import org.mg.wekalib.data.InstanceUtil;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.RegressionByDiscretization;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class RegressionWithClassProb extends TransformDataClassifier
{
	List<Classifier> traininedPropComputers = new ArrayList<>();
	List<Double> discPercentile = ListUtil.createList(90.0, 10.0);

	public RegressionWithClassProb()
	{
	}

	@Override
	public Instances transformData(Instances data, boolean training) throws Exception
	{
		List<List<String>> discVals = new ArrayList<>();
		if (training)
		{
			DescriptiveStatistics stats = new DescriptiveStatistics();
			for (Instance inst : data)
				stats.addValue(inst.classValue());

			for (Double perc : discPercentile)
			{
				List<String> v = new ArrayList<>();
				double thres = stats.getPercentile(perc);
				for (Instance inst : data)
					v.add(inst.classValue() >= thres ? "H" : "L");
				discVals.add(v);
			}
		}
		else
		{
			for (@SuppressWarnings("unused")
			Double perc : discPercentile)
			{
				List<String> v = new ArrayList<>();
				for (int i = 0; i < data.numInstances(); i++)
					v.add(null);
				discVals.add(v);
			}
		}

		//		System.out.println(data);

		List<List<Double>> predictions = new ArrayList<>();

		if (training)
			traininedPropComputers.clear();
		Instances discData = InstanceUtil.stripAttributes(data,
				ListUtil.createList(data.classAttribute()));
		int idx = 0;
		for (List<String> discV : discVals)
		{
			Instances discDataX = new Instances(discData);
			InstanceUtil.attachNominalAttribute(discDataX, "disc", ListUtil.createList("H", "L"),
					discV, true);
			discDataX.setClassIndex(discDataX.numAttributes() - 1);

			//		System.out.println(discData);

			Classifier propComputer;
			if (training)
			{
				propComputer = new RandomForest();
				propComputer.buildClassifier(discDataX);
				traininedPropComputers.add(propComputer);
			}
			else
				propComputer = traininedPropComputers.get(idx++);

			List<Double> preds = new ArrayList<>();
			for (Instance inst : discDataX)
				preds.add(propComputer.distributionForInstance(inst)[0]);
			predictions.add(preds);
		}

		Instances transformedData = new Instances(data);
		int i = 0;
		for (List<Double> preds : predictions)
			InstanceUtil.attachNumericAttribute(transformedData, "prob" + discPercentile.get(i++),
					preds, false);
		transformedData.setClassIndex(transformedData.numAttributes() - 1);

		//		if (training)
		//			System.out.println(transformedData);

		return transformedData;
	}

	public static void main(String[] args) throws Exception
	{
		DescriptiveStatistics stats = new DescriptiveStatistics();

		for (String s : new String[] { "auto93.arff", "autoHorse.arff", "autoMpg.arff",
				"autoPrice.arff", "baskball.arff", "bodyfat.arff", "bolts.arff", "breastTumor.arff",
				"cholesterol.arff", "cleveland.arff", "cloud.arff", "cpu.arff", "detroit.arff",
				"echoMonths.arff", "elusage.arff", "fishcatch.arff", "fruitfly.arff",
				"gascons.arff", "housing.arff", "hungarian.arff", "longley.arff", "lowbwt.arff",
				"mbagrade.arff", "meta.arff", "pbc.arff", "pharynx.arff", "pollution.arff",
				"pwLinear.arff", "quake.arff", "schlvote.arff", "sensory.arff", "servo.arff",
				"sleep.arff", "strike.arff", "veteran.arff", "vineyard.arff" })
		{
			Instances inst = new Instances(
					new FileReader(System.getProperty("user.home") + "/data/weka/numeric/" + s));
			inst.setClassIndex(inst.numAttributes() - 1);
			System.out.println(s);
			System.out.println("instances: " + inst.numInstances());
			System.out.println("attributes: " + inst.numInstances());

			double corr1;
			double corr2;

			{
				Evaluation eval = new Evaluation(inst);
				Classifier regrL = new RegressionByDiscretization();// IBk(3); // new SMOreg
				((RegressionByDiscretization) regrL).setClassifier(new RandomForest());
				eval.crossValidateModel(regrL, inst, 10, new Random(1), new Object[0]);
				//				System.out.println("rmse " + eval.rootMeanSquaredError());
				System.out.println("corr " + eval.correlationCoefficient());
				corr1 = eval.correlationCoefficient();
			}
			{
				Evaluation eval = new Evaluation(inst);
				RegressionWithClassProb regr = new RegressionWithClassProb();
				Classifier regrL = new RegressionByDiscretization();// IBk(3);
				((RegressionByDiscretization) regrL).setClassifier(new RandomForest());
				regr.setClassifier(regrL);
				eval.crossValidateModel(regr, inst, 10, new Random(1), new Object[0]);
				//				System.out.println("rmse " + eval.rootMeanSquaredError());
				System.out.println("corr " + eval.correlationCoefficient());
				corr2 = eval.correlationCoefficient();
			}
			System.out.println((corr2 - corr1) + "\n");

			stats.addValue(corr2 - corr1);
		}

		System.out.println(stats.getMean() + " +- " + stats.getStandardDeviation());
		System.out.println(stats.getPercentile(50));
	}
}
