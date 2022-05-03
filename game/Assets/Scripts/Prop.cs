﻿using System;
using UnityEngine;

public class Prop
{
    private ActionQueue _actionQueue;
    private GameObject _asset;
    private IAssetSource.AssetId _assetId;

    // If set (see SetOutline()), contains a reference to the outline geometry for this prop.
    // The outline geometry is only renderered if the State's BorderRadius is non-zero.
    private GameObject _outline;

    public static Prop FromNetwork(Network.Prop netProp)
    {
        if (netProp.prop_type != Network.PropType.SIMPLE)
        {
            Debug.Log("Warning, attempted to initialize simple prop from non-simple network message.");
            return null;
        }
        UnityAssetSource assetSource = new UnityAssetSource();
        GameObject obj = assetSource.Load((IAssetSource.AssetId)netProp.simple_init.asset_id);
        IAssetSource.AssetId assetId = (IAssetSource.AssetId)netProp.simple_init.asset_id;
        Prop prop = new Prop(obj, assetId);
        prop.AddAction(Init.InitAt(netProp.prop_info.location, netProp.prop_info.rotation_degrees));
        return prop;
    }

    public Prop(GameObject obj, IAssetSource.AssetId assetId)
    {
        _actionQueue = new ActionQueue();
        _assetId = assetId;
        // If the GameObject we've been provided with is a prefab,
        // instantiate it in the game world.
        if (obj.scene.name == null)
        {
            _asset = GameObject.Instantiate(obj, new Vector3(0, 0, 0), Quaternion.identity);
        }
        else
        {
            _asset = obj;
        }
    }

    public void SetScale(float scale)
    {
        _asset.transform.localScale = new Vector3(scale, scale, scale);
    }

    public GameObject Find(string path)
    {
        Transform transform = _asset.transform.Find(path);
        if (transform == null) return null;
        return transform.gameObject;
    }

    public void SetOutline(GameObject outline)
    {
        _outline = outline;
    }

    // Returns true if the actor is in the middle of an action.
    public bool IsBusy() { return _actionQueue.IsBusy(); }

    // Returns the actor's current location (or destination, if busy).
    public HecsCoord Location() { return _actionQueue.TargetState().Coord; }
    
    // Returns the actor's worldspace coordinates.
    public Vector3 Position() { return _asset.transform.position; }

    // Returns the actor's current heading (or destination, if rotating).
    public float HeadingDegrees() { return _actionQueue.TargetState().HeadingDegrees; }

    public void SetParent(GameObject parent)
    {
        _asset.transform.SetParent(parent.transform, false);
    }

    public void SetTag(string tag)
    {
        _asset.tag = tag;
    }

    public void Update()
    {
        UnityAssetSource assetSource = new UnityAssetSource();
        _actionQueue.Update();

        State.Discrete discrete = _actionQueue.State();
        if (discrete.EndOfLife)
        {
            if (!IsDestroyed()) Destroy();
            return;
        }
        if (IsDestroyed()) return;

        State.Continuous state = _actionQueue.ContinuousState();
        // Update current location, orientation, and animation based on action queue.
        _asset.transform.position = Scale() * state.Position;
        _asset.transform.rotation = Quaternion.AngleAxis(state.HeadingDegrees, new Vector3(0, 1, 0));

        // If the object has an outline geometry, conditionally scale and draw it.
        if (_outline != null)
        {
            MeshRenderer renderer = _outline.GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                renderer.enabled = state.BorderRadius > 0;
                float z_scale = 1.0f + (state.BorderRadius / 100);
                float x_scale = 1.0f + (2.0f * state.BorderRadius / 100);
                float height = _outline.transform.localScale.y;
                _outline.transform.localScale = new Vector3(x_scale, height, z_scale);
                Color borderColor = (state.BorderColor != null) ? state.BorderColor.ToUnity() : Color.magenta;
                renderer.material.SetColor("_Color", borderColor);
                renderer.material.SetColor("_EmissionColor", borderColor);
            }
        }

        Animation animation = _asset.GetComponentInChildren<Animation>();
        if (animation == null)
            return;
        if (_assetId == IAssetSource.AssetId.FOLLOWER_BOT)
        {
            // The follower bot animation is always default set to hover. Don't
            // bother adjusting the animation.
            return;
        }
        if (state.Animation == ActionQueue.AnimationType.WALKING)
        {
            animation.Play("Armature|Walking");
        }
        else if (state.Animation == ActionQueue.AnimationType.IDLE)
        {
            // Fade into idle, to remove artifacts if we're in the middle of another animation.
            animation.CrossFade("Armature|Idle", 0.3f);
        }
        else
        {
            // All other animations default to idle.
            animation.CrossFade("Armature|Idle", 0.3f);
        }
    }

    // A prop could self-deallocate if a Death action is sent to it. This allows
    // the user to retrieve the prop's state.
    public bool IsDestroyed() { return _asset == null; }

    // Flushes actions and deallocates the assets for this object.
    public void Destroy()
    {
        GameObject.Destroy(_asset);
        _asset = null;
        if (_outline != null)
        {
            GameObject.Destroy(_outline);
            _outline = null;
        }
    }

    // Flushes actions in flight.
    public void Flush()
    {
        _actionQueue.Flush();
    }

    public void AddAction(ActionQueue.IAction action)
    {
        _actionQueue.AddAction(action);
    }

    private float Scale()
    {
        GameObject obj = GameObject.FindWithTag(HexGrid.TAG);
        HexGrid manager = obj.GetComponent<HexGrid>();
        return manager.Scale;
    }
}
