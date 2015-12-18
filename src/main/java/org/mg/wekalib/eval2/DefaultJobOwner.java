package org.mg.wekalib.eval2;

import java.io.Serializable;

import org.mg.wekalib.eval2.util.ResultProvider;

public abstract class DefaultJobOwner<R extends Serializable> implements JobOwner<R>
{
	@Override
	public boolean isDone()
	{
		return ResultProvider.contains(key());
	}

	@SuppressWarnings("unchecked")
	@Override
	public R getResult()
	{
		return (R) ResultProvider.get(key());
	}

	protected void setResult(R r)
	{
		ResultProvider.set(key(), r);
	}
}
