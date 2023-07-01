﻿using System;
using System.Collections.Generic;
using UnityEngine;

// Utility for handling a HECS-based hexagonal grid.
//
// The neighbor direction functions only make sense if this is a pointy-top 
[Serializable]
public struct HecsCoord
{
    public int a, r, c;

    public static readonly HecsCoord ORIGIN = new HecsCoord(0, 0, 0);

    private static readonly Dictionary<HecsCoord, string> LOC_TO_DIR = new Dictionary<HecsCoord, string>
	{
	    {HecsCoord.ORIGIN.UpRight(), "UP_RIGHT"},
	    {HecsCoord.ORIGIN.Right(), "RIGHT"},
	    {HecsCoord.ORIGIN.DownRight(), "DOWN_RIGHT"},
	    {HecsCoord.ORIGIN.DownLeft(), "DOWN_LEFT"},
	    {HecsCoord.ORIGIN.Left(), "LEFT"},
	    {HecsCoord.ORIGIN.UpLeft(), "UP_LEFT"},
	};

    public string DirectionOf(HecsCoord b)
    {
        HecsCoord displacement = HecsCoord.Sub(b, this);
        if (!LOC_TO_DIR.ContainsKey(displacement)) return "";
        return LOC_TO_DIR[displacement];
    }

    public bool IsAdjacentTo(HecsCoord b)
    {
        HecsCoord displacement = HecsCoord.Sub(b, this);
        return LOC_TO_DIR.ContainsKey(displacement);
    }

    // From offset coordinates. This matches the "odd-r" scheme documented
    // on this page:
    // https://www.redblobgames.com/grids/hexagons/#coordinates
    public static HecsCoord FromOffsetCoordinates(int r, int c)
    {
        return new HecsCoord(r % 2, r / 2, c);
    }

    // Returns offset coordinates [row, col].
    public (int, int) ToOffsetCoordinates()
    {
        return (2 * r + a, c);
    }

    // A 1-D range of Hecs coordinates, all on the same row.
    public static HecsCoord[] Range1D(HecsCoord origin, int cols)
    {
        HecsCoord[] row = new HecsCoord[cols];
        for (int i = 0; i < cols; ++i)
        {
                row[i] = origin;
                origin = origin.Right();
        }
        return row;
    }

    //  2-D range of Hecs coordiantes.
    public static HecsCoord[][] Range2D(HecsCoord origin, int rows, int cols)
    {
        HecsCoord[][] grid = new HecsCoord[rows][];
        for (int i = 0; i < rows/2; ++i)
        {
            HecsCoord[] row = Range1D(origin, cols);
            grid[2*i] = row;
            origin = origin.DownRight();

            row = Range1D(origin, cols);
            grid[2*i + 1] = row;
            origin = origin.DownLeft();
        }
        if (rows % 2 == 1)
        {
            origin = origin.DownRight();
            HecsCoord[] row = Range1D(origin, cols);
            grid[rows - 1] = row;
        }
        return grid;
    }

    public static HecsCoord Add(HecsCoord a, HecsCoord b)
    {
        // https://en.wikipedia.org/wiki/Hexagonal_Efficient_Coordinate_System#Addition
        return new HecsCoord(
            a.a ^ b.a,
			a.r + b.r + (a.a & b.a),
			a.c + b.c + (a.a & b.a));
    }

    public static HecsCoord Sub(HecsCoord a, HecsCoord b)
    {
        return Add(a, b.Negate());
    }

    public HecsCoord(int in_a, int in_r, int in_c) {
        a = in_a;
        r = in_r;
        c = in_c;
    }

    // HecsCoord can be used as the key in a dictionary.
    public override int GetHashCode()
    {
        return (a, r, c).GetHashCode();
    }
    public bool Equals(HecsCoord obj)
    {
        var rhs = (obj.a, obj.r, obj.c);
        var lhs = (a, r, c);
        return lhs == rhs;
    }

    public HecsCoord UpRight()
    {
        return new HecsCoord(1 - a, r - (1 - a), c + a);
    }
    public HecsCoord Right()
    {
        return new HecsCoord(a, r, c + 1);
    }
    public HecsCoord DownRight()
    {
        return new HecsCoord(1 - a, r + a, c + a);
    }
    public HecsCoord DownLeft()
    {
        return new HecsCoord(1 - a, r + a, c - (1 - a));
    }
    public HecsCoord Left()
    {
        return new HecsCoord(a, r, c - 1);
    }
    public HecsCoord UpLeft()
    {
        return new HecsCoord(1 - a, r - (1 - a), c - (1 - a));
    }

    public HecsCoord Negate()
    {
        // https://en.wikipedia.org/wiki/Hexagonal_Efficient_Coordinate_System#Negation
        return new HecsCoord(a, -r - a, -c - a);
    }

    public HecsCoord[] Neighbors()
    {
        // https://en.wikipedia.org/wiki/Hexagonal_Efficient_Coordinate_System#Nearest_neighbors
        return new HecsCoord[]{
            UpRight(),
            Right(),
            DownRight(),
            DownLeft(),
            Left(),
            UpLeft()
        };
    }

    public HecsCoord NeighborAtHeading(float heading)
    {
        int neighborIndex = ((int)(heading / 60.0f)) % 6;
        if (neighborIndex < 0)
            neighborIndex += 6;
        return Neighbors()[neighborIndex];
    }

    public (float, float) Cartesian()
    {
        // https://en.wikipedia.org/wiki/Hexagonal_Efficient_Coordinate_System#Convert_to_Cartesian
        return (0.5f * a + c, - Mathf.Sqrt(3) / 2.0f * a - Mathf.Sqrt(3) * r);
    }

    public override string ToString()
    {
        // Return a fixed-length string of the format (a, r, c). Two digits for each coordinate.
        return "(" + a.ToString("D2") + ", " + r.ToString("D2") + ", " + c.ToString("D2") + ")";
    }
}