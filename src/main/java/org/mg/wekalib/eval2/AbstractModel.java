package org.mg.wekalib.eval2;

import org.mg.wekalib.eval2.util.Blocker;
import org.mg.wekalib.eval2.util.Printer;
import org.mg.wekalib.evaluation.CVPredictionsEvaluation;
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
	public String key()
	{
		StringBuffer b = new StringBuffer();
		b.append(getWekaClassifer().getClass().getSimpleName());
		b.append('#');
		b.append(getParamKey());
		b.append('#');
		b.append(train == null ? null : train.key());
		b.append('#');
		b.append(test == null ? null : test.key());
		return b.toString();
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		if (!Blocker.block(key()))
			return null;
		return new Runnable()
		{
			public void run()
			{
				validateModel();
				Blocker.unblock(AbstractModel.this.key());
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
			Printer.println("building model " + getName() + " on " + trainI.relationName() + " " + key());
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

	public abstract String getParamKey();

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
