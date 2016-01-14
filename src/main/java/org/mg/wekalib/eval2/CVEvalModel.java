package org.mg.wekalib.eval2;

import java.io.File;
import java.io.FileReader;

import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.data.WekaInstancesDataSet;
import org.mg.wekalib.eval2.job.DefaultJobOwner;
import org.mg.wekalib.eval2.job.Printer;
import org.mg.wekalib.eval2.model.Model;
import org.mg.wekalib.eval2.model.NaiveBayesModel;
import org.mg.wekalib.eval2.model.RandomForestModel;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekalib.evaluation.Predictions;

import weka.core.Instances;

/**
 * model, that evaluates the best model with inner cross-validation ({@link CVEvaluator} 
 */
public class CVEvalModel extends DefaultJobOwner<Predictions> implements Model
{
	CVEvaluator cvEval = new CVEvaluator();
	DataSet train;
	DataSet test;

	@Override
	public String getKeyContent()
	{
		return getKeyContent(cvEval, train, test);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		CVEvaluator cv = cvEval.cloneJob();
		cv.setDataSet(train);
		if (!cv.isDone())
			return Printer.wrapRunnable("CVEvalModel: inner CV", cv.nextJob());
		else
		{
			final Model best = (Model) cv.getBestModel().cloneJob();
			best.setTrainingDataset(train);
			best.setTestDataset(test);
			if (!best.isDone())
				return Printer.wrapRunnable("CVEvalModel: build model", best.nextJob());
			else
				return blockedJob("CVEvalModel: storing results", storeResults(best));
		}
	}

	private Runnable storeResults(final Model m)
	{
		return new Runnable()
		{
			public void run()
			{
				Predictions p = m.getResult();
				//System.err.println(PredictionUtil.summaryClassification(p));
				setResult(p);
			};
		};
	}

	@Override
	public Model cloneJob()
	{
		CVEvalModel m = new CVEvalModel();
		m.cvEval = cvEval;
		return m;
	}

	public void setCvEvaluator(CVEvaluator cvEval)
	{
		this.cvEval = cvEval;
	}

	@Override
	public void setTrainingDataset(DataSet train)
	{
		this.train = train;
	}

	@Override
	public void setTestDataset(DataSet test)
	{
		this.test = test;
	}

	@Override
	public DataSet getTrainingDataset()
	{
		return train;
	}

	@Override
	public String getName()
	{
		return "CVEvalModel";
	}

	@Override
	public String getKeyPrefix()
	{
		return "CVEvalModel" + (train != null ? (File.separator + train.getKeyPrefix()) : "");
	}

	@Override
	public String getAlgorithmParamsNice()
	{
		return "";
	}

	@Override
	public String getAlgorithmShortName()
	{
		return getName();
	}

	public static void main(String[] args) throws Exception
	{
		CVEvaluator cv = new CVEvaluator();
		cv.setModels(new RandomForestModel(), new NaiveBayesModel());
		cv.setNumFolds(10);
		cv.setRepetitions(1);

		CVEvalModel cvM = new CVEvalModel();
		cvM.setCvEvaluator(cv);

		Instances inst = new Instances(
				new FileReader("/home/martin/data/weka/nominal/breast-w.arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		cvM.setTrainingDataset(new WekaInstancesDataSet(inst, 1));
		cvM.setTestDataset(new WekaInstancesDataSet(inst, 1));

		cvM.runSequentially();
		System.out.println(PredictionUtil.summaryClassification(cvM.getResult()));
	}

}
