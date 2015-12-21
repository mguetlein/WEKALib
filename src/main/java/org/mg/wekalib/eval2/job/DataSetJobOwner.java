package org.mg.wekalib.eval2.job;

import java.io.Serializable;

import org.mg.wekalib.eval2.data.DataSet;

public interface DataSetJobOwner<R extends Serializable> extends JobOwner<R>
{
	public void setDataSet(DataSet d);
}
