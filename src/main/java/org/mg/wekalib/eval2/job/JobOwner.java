package org.mg.wekalib.eval2.job;

import java.io.Serializable;

public interface JobOwner<R extends Serializable> extends KeyProvider, ComposedKeyProvider
{
	public String getName();

	public boolean isDone();

	public R getResult();

	public JobOwner<R> cloneJob();

	public Runnable nextJob() throws Exception;

	public R runSequentially() throws Exception;

	public R runParrallel(int numThreads) throws Exception;
}
