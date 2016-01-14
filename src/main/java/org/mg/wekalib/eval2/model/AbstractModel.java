package org.mg.wekalib.eval2.model;

import java.io.File;

import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.job.DefaultJobOwner;
import org.mg.wekalib.evaluation.CVPredictionsEvaluation;
import org.mg.wekalib.evaluation.Predictions;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.NonSparseToSparse;

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
		return getWekaClassifer().getClass().getSimpleName();
	}

	@Override
	public String getKeyPrefix()
	{
		return getWekaClassifer().getClass().getSimpleName()
				+ (train != null ? (File.separator + train.getName()) : "");
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
			if (classifier instanceof SMO)
			{
				//				System.err.print("filtering..");
				NonSparseToSparse filter = new NonSparseToSparse();
				filter.setInputFormat(trainI);
				trainI = Filter.useFilter(trainI, filter);
				testI = Filter.useFilter(testI, filter);
				//				System.err.println("..done");
			}
			classifier.buildClassifier(trainI);
			CVPredictionsEvaluation eval = new CVPredictionsEvaluation(trainI);
			eval.evaluateModel(classifier, testI);
			Predictions p = eval.getCvPredictions();
			//			System.err.println(PredictionUtil.summaryClassification(p));
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

}
