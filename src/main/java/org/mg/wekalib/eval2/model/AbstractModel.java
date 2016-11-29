package org.mg.wekalib.eval2.model;

import java.io.File;

import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.data.SplitDataSet;
import org.mg.wekalib.eval2.job.DefaultJobOwner;
import org.mg.wekalib.evaluation.Predictions;
import org.mg.wekalib.evaluation.PredictionsEvaluation;

import weka.classifiers.Classifier;
import weka.core.Instances;

public abstract class AbstractModel extends DefaultJobOwner<Predictions> implements Model
{
	protected DataSet train;
	protected DataSet test;

	@Override
	public Model cloneJob()
	{
		try
		{
			Model m = this.getClass().newInstance();
			cloneParams(m);
			m.setTestDataset(test);
			m.setTrainingDataset(train);
			return m;
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	@Override
	public String getName()
	{
		return getWekaClassifierName();
	}

	public String getWekaClassifierName()
	{
		return getWekaClassifer().getClass().getSimpleName();
	}

	@Override
	public String getKeyPrefix()
	{
		String prefix = "";
		if (train != null)
			prefix += train.getKeyPrefix() + File.separator;
		prefix += getWekaClassifierName();
		return prefix;
	}

	@Override
	public String getKeyContent()
	{
		return getKeyContent(getParamKey(), train, test);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		return blockedJob("Model: building " + getName() + " on " + train.getName(), new Runnable()
		{
			public void run()
			{
				validateModel();
			};
		});
	}

	private void validateModel()
	{
		try
		{
			Classifier classifier = getWekaClassifer();
			Instances trainI = train.getWekaInstances();
			Instances testI = test.getWekaInstances();

			classifier.buildClassifier(trainI);

			PredictionsEvaluation eval = new PredictionsEvaluation(trainI);
			eval.evaluateModel(classifier, testI);

			Predictions p = eval.getCvPredictions();

			if (test instanceof SplitDataSet)
				for (int i = 0; i < p.origIndex.length; i++)
					p.origIndex[i] = ((SplitDataSet) test).getOrigIndex(i);

			// System.err.println(PredictionUtil.summaryClassification(p));
			// PredictionUtil.printPredictionsWithConfidence(p, train.getPositiveClass());

			setResult(p);
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

	public abstract Classifier getWekaClassifer();

	public abstract boolean isFast();

	protected abstract String getParamKey();

	protected abstract void cloneParams(Model clonedModel);

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
	public boolean isValid(DataSet dataSet)
	{
		return true;
	}

}
