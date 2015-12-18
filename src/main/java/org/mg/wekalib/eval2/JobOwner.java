package org.mg.wekalib.eval2;

import java.io.Serializable;

public interface JobOwner<R extends Serializable>
{
	public String key();

	public boolean isDone();

	public Runnable nextJob() throws Exception;

	public R getResult();
}
