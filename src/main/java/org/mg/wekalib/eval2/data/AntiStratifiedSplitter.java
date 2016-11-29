package org.mg.wekalib.eval2.data;

import java.util.List;

import org.mg.wekalib.eval2.job.ComposedKeyProvider;

public interface AntiStratifiedSplitter extends ComposedKeyProvider
{
	public List<Integer> antiStratifiedSplitIndices(DataSet dataset, double ratio, long seed,
			boolean train);
}
