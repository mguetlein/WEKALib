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
 * model, that evaluates the best model with inner cross-validation ({@link ValidationEval} 
 */
public class ValidatedModel extends DefaultJobOwner<Predictions> implements Model
{
	ValidationEval validationEval = new ValidationEval();
	DataSet train;
	DataSet test;

	@Override
	public String getKeyContent()
	{
		return getKeyContent(validationEval, train, test);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		ValidationEval v = validationEval.cloneJob();
		v.setDataSet(train);
		if (!v.isDone())
			return Printer.wrapRunnable(getName() + ": inner Evaluation", v.nextJob());
		else
		{
			final Model best = (Model) v.getBestModel().cloneJob();
			best.setTrainingDataset(train);
			best.setTestDataset(test);
			if (!best.isDone())
				return Printer.wrapRunnable(getName() + ": build model", best.nextJob());
			else
				return blockedJob(getName() + ": storing results", storeResults(best));
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
		ValidatedModel m = new ValidatedModel();
		m.validationEval = validationEval;
		return m;
	}

	public void setValidationEvaluator(ValidationEval validationEval)
	{
		this.validationEval = validationEval;
	}

	public ValidationEval getValidationEvaluator()
	{
		return validationEval;
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
		return this.getClass().getSimpleName();
	}

	@Override
	public String getKeyPrefix()
	{
		String prefix = "";
		if (train != null)
			prefix += train.getKeyPrefix() + File.separator;
		prefix += getName();
		return prefix;
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
		//		CV cv = new CV();
		//		cv.setNumFolds(10);
		Holdout ho = new Holdout();
		ho.setSplitRatio(0.7);

		ValidationEval eval = new ValidationEval();
		eval.setModels(new RandomForestModel(), new NaiveBayesModel());
		eval.setValidation(ho);
		//cv.setNumFolds(10);
		eval.setRepetitions(1);

		ValidatedModel evalM = new ValidatedModel();
		evalM.setValidationEvaluator(eval);

		Instances inst = new Instances(
				new FileReader("/home/martin/data/weka/nominal/breast-w.arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		evalM.setTrainingDataset(new WekaInstancesDataSet(inst, 1));
		evalM.setTestDataset(new WekaInstancesDataSet(inst, 1));

		evalM.runSequentially();
		System.out.println(PredictionUtil.summaryClassification(evalM.getResult()));
	}

	@Override
	public boolean isValid(DataSet dataSet)
	{
		return true;
	}
}
