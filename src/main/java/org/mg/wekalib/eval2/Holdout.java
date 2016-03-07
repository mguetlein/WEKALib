package org.mg.wekalib.eval2;

import java.io.File;
import java.io.FileReader;

import org.mg.wekalib.eval2.data.WekaInstancesDataSet;
import org.mg.wekalib.eval2.job.Printer;
import org.mg.wekalib.eval2.model.Model;
import org.mg.wekalib.eval2.model.RandomForestModel;
import org.mg.wekalib.evaluation.PredictionUtil;

import weka.core.Instances;

public class Holdout extends Validation
{
	double splitRatio;

	public Holdout cloneJob()
	{
		Holdout ho = new Holdout();
		ho.setDataSet(dataSet);
		ho.setModel(model);
		ho.setSplitRatio(splitRatio);
		ho.setRandomSeed(randomSeed);
		ho.setStratified(stratified);
		return ho;
	}

	@Override
	public String getName()
	{
		return "Holdout: splitRatio " + splitRatio + ", seed " + randomSeed;
	}

	private Model getHoldoutModel()
	{
		Model m = (Model) model.cloneJob();
		m.setTrainingDataset(dataSet.getTrainSplit(splitRatio, stratified, randomSeed));
		m.setTestDataset(dataSet.getTestSplit(splitRatio, stratified, randomSeed));
		return m;
	}

	@Override
	public String getKeyPrefix()
	{
		if (dataSet != null)
			return getHoldoutModel().getKeyPrefix();
		else
			return "Holdout-splitRatio" + splitRatio + "-seed" + randomSeed + "-strat" + stratified
					+ File.separator + model.getKeyPrefix();
	}

	@Override
	public String getKeyContent()
	{
		if (dataSet != null)
			return getHoldoutModel().getKeyContent();
		else
			return getKeyContent(model, splitRatio, randomSeed, stratified);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		if (dataSet == null)
			throw new NullPointerException("set dataset first");

		return Printer.wrapRunnable("Holdout: ratio " + splitRatio + ", seed " + randomSeed,
				getHoldoutModel().nextJob());
	}

	public void setSplitRatio(double splitRatio)
	{
		this.splitRatio = splitRatio;
	}

	public static void main(String[] args) throws Exception
	{
		Holdout ho = new Holdout();
		Instances inst = new Instances(
				new FileReader("/home/martin/data/weka/nominal/breast-w.arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		ho.setDataSet(new WekaInstancesDataSet(inst, 1));
		ho.setModel(new RandomForestModel());
		ho.setSplitRatio(0.7);
		ho.runSequentially();
		System.out.println(PredictionUtil.summaryClassification(ho.getResult()));

	}

}
