package org.mg.wekalib.eval2;

import org.mg.wekalib.eval2.util.Blocker;
import org.mg.wekalib.eval2.util.Printer;
import org.mg.wekautil.Predictions;

public class CVEvalModel extends DefaultJobOwner<Predictions> implements Model
{
	private static final long serialVersionUID = 1L;

	CVEvaluator cvEval = new CVEvaluator();
	DataSet train;
	DataSet test;

	@Override
	public String key()
	{
		StringBuffer b = new StringBuffer();
		b.append(cvEval.key());
		b.append('#');
		b.append(train == null ? null : train.key());
		b.append('#');
		b.append(test == null ? null : test.key());
		return b.toString();
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		CVEvaluator cv = cvEval.cloneCVEvaluator();
		cv.setDataSet(train);
		if (!cv.isDone())
			return Printer.wrapRunnable("CVEvalModel: inner CV", cv.nextJob());

		final Model best = cv.getBestModel().cloneModel();
		best.setTrainingDataset(train);
		best.setTestDataset(test);
		if (!best.isDone())
			return Printer.wrapRunnable("CVEvalModel: build model", best.nextJob());

		if (!Blocker.block(key()))
			return null;
		return new Runnable()
		{
			public void run()
			{
				Printer.println("CVEvalModel: store model result " + CVEvalModel.this.key());
				Predictions p = best.getResult();
				//System.err.println(PredictionUtil.summaryClassification(p));
				setResult(p);
				Blocker.unblock(CVEvalModel.this.key());
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
