package org.mg.wekalib.eval2;

import java.io.File;
import java.io.FileReader;

import org.mg.wekalib.eval2.data.AntiStratifiedSplitter;
import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.data.EuclideanPCAWekaAntiStratifiedSplitter;
import org.mg.wekalib.eval2.data.WekaInstancesDataSet;
import org.mg.wekalib.eval2.job.Printer;
import org.mg.wekalib.eval2.model.Model;
import org.mg.wekalib.eval2.model.RandomForestModel;
import org.mg.wekalib.evaluation.PredictionUtil;

import weka.core.Instances;

public class Holdout extends Validation
{
	double splitRatio;
	AntiStratifiedSplitter antiStratifiedSplitter;

	public Holdout cloneJob()
	{
		Holdout ho = new Holdout();
		ho.setDataSet(dataSet);
		ho.setModel(model);
		ho.setSplitRatio(splitRatio);
		ho.setRandomSeed(randomSeed);
		ho.setStratified(stratified);
		ho.setAntiStratifiedSplitter(antiStratifiedSplitter);
		return ho;
	}

	@Override
	public String getName()
	{
		return "Holdout: splitRatio " + splitRatio + ", seed " + randomSeed + ", antiStrat "
				+ antiStratifiedSplitter;
	}

	private Model getHoldoutModel()
	{
		Model m = (Model) model.cloneJob();
		m.setTrainingDataset(
				dataSet.getTrainSplit(splitRatio, stratified, randomSeed, antiStratifiedSplitter));
		m.setTestDataset(
				dataSet.getTestSplit(splitRatio, stratified, randomSeed, antiStratifiedSplitter));
		return m;
	}

	public DataSet getTrainingDataSet()
	{
		return dataSet.getTrainSplit(splitRatio, stratified, randomSeed, antiStratifiedSplitter);
	}

	public DataSet getTestDataSet()
	{
		return dataSet.getTestSplit(splitRatio, stratified, randomSeed, antiStratifiedSplitter);
	}

	@Override
	public String getKeyPrefix()
	{
		if (dataSet != null)
			return getHoldoutModel().getKeyPrefix();
		else if (antiStratifiedSplitter == null)
			return "Holdout-splitRatio" + splitRatio + "-seed" + randomSeed + "-strat" + stratified
					+ File.separator + model.getKeyPrefix();
		else
			return "Holdout-splitRatio" + splitRatio + "-seed" + randomSeed + "-strat" + stratified
					+ "-anti" + antiStratifiedSplitter + File.separator + model.getKeyPrefix();
	}

	@Override
	public String getKeyContent()
	{
		if (dataSet != null)
			return getHoldoutModel().getKeyContent();
		else if (antiStratifiedSplitter == null)
			return getKeyContent(model, splitRatio, randomSeed, stratified);
		else
			return getKeyContent(model, splitRatio, randomSeed, stratified, antiStratifiedSplitter);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		if (dataSet == null)
			throw new NullPointerException("set dataset first");

		return Printer.wrapRunnable("Holdout: ratio " + splitRatio + ", seed " + randomSeed
				+ ", antiStrat " + antiStratifiedSplitter, getHoldoutModel().nextJob());
	}

	public void setSplitRatio(double splitRatio)
	{
		this.splitRatio = splitRatio;
	}

	public void setAntiStratifiedSplitter(AntiStratifiedSplitter antiStratifiedSplitter)
	{
		this.antiStratifiedSplitter = antiStratifiedSplitter;
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
		ho.setAntiStratifiedSplitter(new EuclideanPCAWekaAntiStratifiedSplitter());
		ho.runSequentially();
		System.out.println(PredictionUtil.summaryClassification(ho.getResult()));

	}

}
