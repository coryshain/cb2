﻿using System;
using UnityEngine;

public class Translate : HexMovement.IAnimation
{
    private HexMovement.AnimationInfo _info;
    private Vector3 _location;
    private DateTime _start;

    public Translate(HexMovement.AnimationInfo info)
    {
    }

    public void Start()
    {
        _start = DateTime.Now;
    }

    public HexMovement.AnimationInfo Info() { return _info;  }

    public void Update()
    {
        float progress =
	        (float)((DateTime.Now - _start).TotalSeconds / _info.DurationS);
        (float sx, float sz) = _info.Start.Cartesian();
        (float dx, float dz) = _info.Destination.Cartesian();
        // I'm going to need to get the ground location...
        Vector3 startLocation = new Vector3(sx, 0, sz);
        Vector3 destinationLocation = new Vector3(dx, 0, dz);
        _location = Vector3.Lerp(startLocation, destinationLocation, progress);
    }

    public float Heading()
    {
        return _info.StartHeading;
    }

    public Vector3 Location()
    {
        return _location;
    }

    public bool IsDone()
    {
        if ((DateTime.Now - _start).TotalSeconds > _info.DurationS) return true;
        if (DateTime.Now > _info.Expiration) return true;
        return false;
    }
}
