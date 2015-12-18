package org.mg.wekalib.eval2;

import org.mg.wekalib.eval2.util.Blocker;
import org.mg.wekalib.eval2.util.Printer;
import org.mg.wekautil.Predictions;

public class FeatureModel extends DefaultJobOwner<Predictions> implements Model
{
	private static final long serialVersionUID = 1L;

	FeatureProvider featureProvider;
	Model model;
	DataSet train;
	DataSet test;

	@Override
	public String key()
	{
		StringBuffer b = new StringBuffer();
		b.append(featureProvider.key());
		b.append('#');
		b.append(model.key());
		b.append('#');
		b.append(train == null ? null : train.key());
		b.append('#');
		b.append(test == null ? null : test.key());
		return b.toString();
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		FeatureProvider feat = featureProvider.cloneFeatureProvider();
		feat.setTrainingDataset(train);
		feat.setTestDataset(test);
		if (!feat.isDone())
			return Printer.wrapRunnable("FeatureModel: compute features", feat.nextJob());

		final Model mod = model.cloneModel();
		DataSet res[] = feat.getResult();
		mod.setTrainingDataset(res[0]);
		mod.setTestDataset(res[1]);
		if (!mod.isDone())
			return Printer.wrapRunnable("FeatureModel: build model", mod.nextJob());

		if (!Blocker.block(key()))
			return null;
		return new Runnable()
		{
			@Override
			public void run()
			{
				Printer.println("FeatureModel: storing results " + FeatureModel.this.key());
				Predictions p = mod.getResult();
				//				System.err.println(PredictionUtil.summaryClassification(p));
				setResult(p);
				Blocker.unblock(FeatureModel.this.key());
			}
		};
	}

	@Override
	public Model cloneModel()
	{
		FeatureModel fm = new FeatureModel();
		fm.featureProvider = featureProvider.cloneFeatureProvider();
		fm.model = model.cloneModel();
		fm.train = train;
		fm.test = test;
		return fm;
	}

	public void setModel(Model model)
	{
		this.model = model;
	}

	public void setFeatureProvider(FeatureProvider featureProvider)
	{
		this.featureProvider = featureProvider;
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
		return "FeatureModel (" + featureProvider.getName() + ", " + model.getName() + ")";
	}

}
