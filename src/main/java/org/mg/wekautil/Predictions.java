package org.mg.wekautil;

import java.io.Serializable;

// is here for backwards prob reasons, to be moved to org.mg.wekalib.evaluation

public class Predictions implements Serializable
{
	public static long serialVersionUID = 4L;

	public double actual[];
	public double predicted[];
	public double confidence[];
	public int fold[];
	public int origIndex[];
}
