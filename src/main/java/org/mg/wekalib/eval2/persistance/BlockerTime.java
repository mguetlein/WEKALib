package org.mg.wekalib.eval2.persistance;

import org.mg.javalib.util.StopWatchUtil;

public class BlockerTime implements Blocker
{
	Blocker b;

	public BlockerTime(Blocker b)
	{
		this.b = b;
	}

	@Override
	public boolean isBlockedByThread(String key, String threadId)
	{
		StopWatchUtil.start("isBlockedByThread");
		boolean b = this.b.isBlockedByThread(key, threadId);
		StopWatchUtil.stop("isBlockedByThread");
		return b;
	}

	@Override
	public boolean block(String key, String threadId)
	{
		StopWatchUtil.start("block");
		boolean b = this.b.block(key, threadId);
		StopWatchUtil.stop("block");
		return b;
	}

	@Override
	public void unblock(String key)
	{
		StopWatchUtil.start("unblock");
		b.unblock(key);
		StopWatchUtil.stop("unblock");
	}

	@Override
	public void clear()
	{
		b.clear();
	}

}
