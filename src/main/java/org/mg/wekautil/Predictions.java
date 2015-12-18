package org.mg.wekautil;

import java.io.Serializable;

// is here for backwards prob reasons, to be moved to org.mg.wekalib.evaluation

public class Predictions implements Serializable
{
	public static final long serialVersionUID = 4156859784481976748L;

	public double actual[];
	public double predicted[];
	public double confidence[];
	public int fold[];
	public int origIndex[];
}
