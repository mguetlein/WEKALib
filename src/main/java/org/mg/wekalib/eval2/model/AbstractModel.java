package org.mg.wekalib.eval2.model;

import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.job.DefaultJobOwner;
import org.mg.wekalib.evaluation.CVPredictionsEvaluation;
import org.mg.wekautil.Predictions;

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
	public String getKey()
	{
		return getKey(getWekaClassifer().getClass().getSimpleName(), getParamKey(), train, test);
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

	public String getName()
	{
		return getWekaClassifer().getClass().getSimpleName();
	}

	public abstract Classifier getWekaClassifer();

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

}
