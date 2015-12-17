package org.mg.wekalib.eval2;

import org.mg.javalib.util.HashUtil;
import org.mg.wekalib.evaluation.CVPredictionsEvaluation;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekautil.Predictions;

import weka.classifiers.Classifier;
import weka.core.Instances;

public abstract class AbstractModel extends DefaultJobOwner<Predictions> implements Model
{
	private static final long serialVersionUID = 1L;

	protected DataSet train;
	protected DataSet test;

	@Override
	public Model cloneModel()
	{
		try
		{
			Model m = this.getClass().newInstance();
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
	public int hashCode()
	{
		return HashUtil.hashCode(getWekaClassifer().getClass().getSimpleName(), getParamKey(), train, test);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		if (!Blocker.block(hashCode()))
			return null;
		return new Runnable()
		{
			public void run()
			{
				validateModel();
				Blocker.unblock(AbstractModel.this.hashCode());
			};
		};
	}

	private void validateModel()
	{
		try
		{
			Classifier classifier = getWekaClassifer();
			Instances trainI = train.getWekaInstances();
			Instances testI = test.getWekaInstances();
			System.out.println(hashCode() + " building model " + getName() + " on " + trainI.relationName() + " key: "
					+ hashCode());
			classifier.buildClassifier(trainI);
			CVPredictionsEvaluation eval = new CVPredictionsEvaluation(trainI);
			eval.evaluateModel(classifier, testI);
			Predictions p = eval.getCvPredictions();
			System.err.println(PredictionUtil.summaryClassification(p));
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

	public abstract int getParamKey();

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
