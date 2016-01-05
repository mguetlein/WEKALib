package org.mg.wekalib.eval2.persistance;

import java.io.Serializable;

import org.mg.javalib.util.StopWatchUtil;

public class ResultProviderTime implements ResultProvider
{
	ResultProvider r;

	public ResultProviderTime(ResultProvider r)
	{
		this.r = r;
	}

	@Override
	public boolean contains(String key)
	{
		StopWatchUtil.start("contains");
		boolean b = r.contains(key);
		StopWatchUtil.stop("contains");
		return b;
	}

	@Override
	public Serializable get(String key)
	{
		StopWatchUtil.start("get");
		Serializable s = r.get(key);
		StopWatchUtil.stop("get");
		return s;
	}

	@Override
	public void set(String key, Serializable value)
	{
		StopWatchUtil.start("set");
		r.set(key, value);
		StopWatchUtil.stop("set");
	}

	@Override
	public void clear()
	{
		r.clear();
	}

}
