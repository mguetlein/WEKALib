package org.mg.wekalib.eval2;

import org.mg.javalib.util.HashUtil;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekautil.Predictions;

public class CVEvalModel extends DefaultJobOwner<Predictions> implements Model
{
	private static final long serialVersionUID = 1L;

	CVEvaluator cvEval = new CVEvaluator();
	DataSet train;
	DataSet test;

	@Override
	public int hashCode()
	{
		return HashUtil.hashCode(cvEval, train, test);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		CVEvaluator cv = cvEval.cloneCVEvaluator();
		cv.setDataSet(train);
		if (!cv.isDone())
			return cv.nextJob();

		final Model best = cv.getBestModel().cloneModel();
		best.setTrainingDataset(train);
		best.setTestDataset(test);
		if (!best.isDone())
			return best.nextJob();

		if (!Blocker.block(hashCode()))
			return null;
		return new Runnable()
		{
			public void run()
			{
				System.out.println(CVEvalModel.this.hashCode() + " store cv eval model result");
				Predictions p = best.getResult();
				System.err.println(PredictionUtil.summaryClassification(p));
				setResult(p);
				Blocker.unblock(CVEvalModel.this.hashCode());
			};
		};
	}

	@Override
	public Model cloneModel()
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
	public String getName()
	{
		return "CVEvalModel";
	}

}
