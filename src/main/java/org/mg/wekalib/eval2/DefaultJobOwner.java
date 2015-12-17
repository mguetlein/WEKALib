package org.mg.wekalib.eval2;

import java.io.Serializable;

public abstract class DefaultJobOwner<R extends Serializable> implements JobOwner<R>
{
	@Override
	public boolean isDone()
	{
		return ResultProvider.contains(hashCode());
	}

	@SuppressWarnings("unchecked")
	@Override
	public R getResult()
	{
		return (R) ResultProvider.get(hashCode());
	}

	protected void setResult(R r)
	{
		ResultProvider.set(hashCode(), r);
	}
}
