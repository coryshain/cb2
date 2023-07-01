using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

public class Player : MonoBehaviour
{
    public static string TAG = "Player";

    public float TurnSpeed = 2;  // Turns/Second.
    public float MoveSpeed = 0.2f;  // Cells/Second.

    public bool ForceStartingPosition = false;
    public int StartingRow = 9;
    public int StartingCol = 7;
    public bool ShowHeading = true;

    private Network.NetworkManager _network;

    private Actor _actor;

    public GameObject OverheadCamera;
    public GameObject AngledOverheadCamera;
    private Camera _fpvCamera;
    private DateTime _lastCameraToggle;
    private Network.TurnState _currentTurn;

    private int _playerId = -1;

    private Logger _logger;

    public static Player TaggedInstance()
    {
        GameObject obj = GameObject.FindGameObjectWithTag(Player.TAG);
        if (obj == null)
            return null;
        return obj.GetComponent<Player>();
    }


    public void SetAssetId(int id)
    {
        // If the ID is the same, don't do anything.
        if (_actor != null)
        {
            if (_actor.AssetId() == (IAssetSource.AssetId)id)
            {
                return;
            }
        }
        UnityAssetSource assets = new UnityAssetSource();
        IAssetSource.AssetId assetId = (IAssetSource.AssetId)id;
        Actor actor = new Actor(assets.Load(assetId), assetId);
        if (_actor != null)
        {
            actor.AddAction(Init.InitAt(_actor.Location(), 0));
            _actor.Destroy();
        }
        _actor = actor;
        _actor.SetParent(gameObject);

        if (_network.Role() == Network.Role.LEADER)
        {
            _actor.SetScale(1.4f);
            _logger.Info("Player animation speed increased.");
            _actor.SetWalkSpeed(1.3f);
        }

        GameObject cameraObj = _actor.Find("Parent/Main Camera");
        if (cameraObj != null)
        {
            _fpvCamera = cameraObj.GetComponent<Camera>();
        }
        InitCamera();

        // If this lobby has standing/vertical cards enabled, make all of the
        // cards "look at" the player. If we're the follower, that is.
        // Fetch lobbyinfo from config to see if cards should stand up and track the follower.
        Network.LobbyInfo lobbyInfo = Network.NetworkManager.TaggedInstance().ServerLobbyInfo();
        Network.Role role = Network.NetworkManager.TaggedInstance().Role();
        if ((lobbyInfo != null) && lobbyInfo.cards_face_follower && (role == Network.Role.FOLLOWER))
        {
            // Iterate through all entitymanager props.
            EntityManager em = EntityManager.TaggedInstance();
            foreach (Prop prop in em.Props())
            {
                // If the prop was already looking at an object, update it to
                // look at the player. In the future if we have multiple lookat
                // targets, this logic will need to be updated to only update
                // cards which are looking at player.
                if (prop.LookAtTarget() != null)
                {
                    prop.SetLookAtTarget(_actor.GetGameObject());
                }
            }
        }
    }


    public void Awake()
    {
        _logger = Logger.GetOrCreateTrackedLogger(TAG);
        UnityAssetSource assets = new UnityAssetSource();
        IAssetSource.AssetId assetId = IAssetSource.AssetId.PLAYER_WITH_CAM;
        _actor = new Actor(assets.Load(assetId), assetId);
        _actor.SetParent(gameObject);

        GameObject obj = GameObject.FindGameObjectWithTag(Network.NetworkManager.TAG);
        _network = obj.GetComponent<Network.NetworkManager>();

        if (_network.Role() == Network.Role.LEADER)
        {
            _actor.SetScale(1.4f);
        }

        if (OverheadCamera == null)
        {
            Debug.Log("Error: No overhead camera provided.");
        }
    }

    void InitCamera()
    {
        GameObject cameraObj = _actor.Find("Parent/Main Camera");
        if (cameraObj != null)
        {
            _fpvCamera = cameraObj.GetComponent<Camera>();
        }
        if ((_network.ServerConfig() != null)
            && (_fpvCamera != null)
            && (_network.Role() == Network.Role.FOLLOWER))
        {
            if (_network.ServerConfig().card_covers)
            {
                _fpvCamera.cullingMask |= 1 << LayerMask.NameToLayer("card_covers");
            } else {
                _fpvCamera.cullingMask &= ~(1 << LayerMask.NameToLayer("card_covers"));
            }
        } else {
            // Probably not necessary... this block of logic could probably be cleaner.
            if (_fpvCamera != null)
                _fpvCamera.cullingMask &= ~(1 << LayerMask.NameToLayer("card_covers"));
        }
        if ((OverheadCamera != null) && (_network.Role() == Network.Role.LEADER))
        {
            if (_fpvCamera != null) _fpvCamera.enabled = false;
            OverheadCamera.GetComponent<Camera>().enabled = false;
            AngledOverheadCamera.GetComponent<Camera>().enabled = true;
        }
        _lastCameraToggle = DateTime.UtcNow;
    }

    public Prop GetProp()
    {
        return _actor.GetProp();
    }

    void Start()
    {
        if (ForceStartingPosition)
        {
            _actor.AddAction(
                Init.InitAt(
                    HecsCoord.FromOffsetCoordinates(StartingRow,
                                                    StartingCol), 0));
        }
        InitCamera();
    }

    public void FlushActionQueue()
    {
        _actor.Flush();
    }

    public void AddAction(ActionQueue.IAction action)
    {
        _actor.AddAction(action);
    }
    public void HandleTurnState(Network.TurnState state)
    {
        _currentTurn = state;
    }

    // Actions are looped back from the server. This method is called to
    // validate actions. If an unrecognized action is received, then 
    // the client can request a state sync from the server.
    public void ValidateHistory(ActionQueue.IAction action)
    {
        // TODO(sharf): Implement this...
    }

    public Vector3 Position()
    {
        return _actor.Position();
    }

    public HecsCoord Coordinate()
    {
        return _actor.Location();
    }

    public float HeadingDegrees()
    {
        return _actor.HeadingDegrees();
    }

    public void SetPlayerId(int playerId) { _playerId = playerId; }
    public int PlayerId() { return _playerId; }

    public void ToggleCameraIfAllowed()
    {
            if ((_network.Role() != Network.Role.LEADER)
                || (OverheadCamera == null)
                || ((DateTime.UtcNow - _lastCameraToggle).TotalMilliseconds <= 500))
                return;
            if (OverheadCamera.GetComponent<Camera>().enabled)
            {
                OverheadCamera.GetComponent<Camera>().enabled = false;
                AngledOverheadCamera.GetComponent<Camera>().enabled = true;
            }
            else
            {
                OverheadCamera.GetComponent<Camera>().enabled = true;
                AngledOverheadCamera.GetComponent<Camera>().enabled = false;
            }
            
            _lastCameraToggle = DateTime.UtcNow;

            // In the tutorial, this hooks camera events to the tutorial manager.
            if (TutorialManager.TaggedInstance() != null)
                TutorialManager.TaggedInstance().CameraToggled();
    }

    void Update()
    {
        if (ShowHeading)
        {
            _actor.EnableDebugging();
        }
        else
        {
            _actor.DisableDebugging();
        }

        _actor.Update();

        // If we're doing an action, don't check for user input.
        if (_actor.IsBusy()) return;

        GameObject obj = GameObject.FindWithTag(HexGrid.TAG);
        HexGrid grid = obj.GetComponent<HexGrid>();

        HecsCoord forwardLocation = _actor.Location().NeighborAtHeading(_actor.HeadingDegrees());
        HecsCoord backLocation = _actor.Location().NeighborAtHeading(_actor.HeadingDegrees() + 180);

        // When a UI element is selected, ignore keypresses. This prevents the
        // player from moving when the user is typing and hits the left or right
        // keys to move the cursor.
        if (EventSystem.current.currentSelectedGameObject != null)
        {
            return;
        }

        if (CameraKey()) ToggleCameraIfAllowed();

        // Don't move until we've received a TurnState.
        if (_currentTurn == null) return;


        // Ignore keypresses when it's not our turn.
        if (_currentTurn.turn != _network.Role())
        {
            return;
        }

        // Don't try to move if we're out of moves.
        if (_currentTurn.moves_remaining <= 0)
        {
            return;
        }

        if (UpKey() &&
            !grid.EdgeBetween(_actor.Location(), forwardLocation))
        {
            Translate action = Translate.Walk(
                HecsCoord.ORIGIN.NeighborAtHeading(_actor.HeadingDegrees()),
                                                   1 / MoveSpeed);
            if (_network.TransmitAction(action)) {
                _actor.AddAction(action);
            }
            return;
        }
        if (DownKey() &&
            !grid.EdgeBetween(_actor.Location(), backLocation))
        {
            Translate action = Translate.Walk(
                HecsCoord.ORIGIN.NeighborAtHeading(
                    _actor.HeadingDegrees() + 180), 1 / MoveSpeed);
            if (_network.TransmitAction(action)) {
                _actor.AddAction(action);
            }
            return;
        }
        if (LeftKey())
        {
            Debug.Log("Heading: " + (_actor.HeadingDegrees() - 60.0f));
            Rotate action = Rotate.Turn(-60.0f, 1 / TurnSpeed);
            if (_network.TransmitAction(action)) {
                _actor.AddAction(action);
            }
            return;
        }
        if (RightKey())
        {
            Rotate action = Rotate.Turn(60.0f, 1 / TurnSpeed);
            if (_network.TransmitAction(action)) {
                _actor.AddAction(action);
            }
            return;
        }
    }

    private bool CameraKey()
    {
        bool keyboard = Input.GetKey(KeyCode.C);
        return keyboard;
    }

    private bool UpKey()
    {
        bool keyboard = Input.GetKey(KeyCode.UpArrow);
        // bool gamepad = Input.GetAxis("Axis 6") < -0.2;
        // bool gamepad_dpad = Input.GetAxis("Axis 10") < -0.2;
        return keyboard;  // || gamepad || gamepad_dpad;
    }

    private bool DownKey()
    {
        bool keyboard = Input.GetKey(KeyCode.DownArrow);
        // bool gamepad = Input.GetAxis("Axis 6") > 0.2;
        // bool gamepad_dpad = Input.GetAxis("Axis 10") > 0.2;
        return keyboard;  // || gamepad || gamepad_dpad;
    }

    private bool LeftKey()
    {
        bool keyboard = Input.GetKey(KeyCode.LeftArrow);
        // bool gamepad = Input.GetAxis("Axis 5") < -0.2;
        // bool gamepad_dpad = Input.GetAxis("Axis 9") < -0.2;
        return keyboard; // || gamepad || gamepad_dpad;
    }

    private bool RightKey()
    {
        bool keyboard = Input.GetKey(KeyCode.RightArrow);
        // bool gamepad = Input.GetAxis("Axis 5") > 0.2;
        // bool gamepad_dpad = Input.GetAxis("Axis 9") > 0.2;
        return keyboard;  // || gamepad || gamepad_dpad;
    }

    private float Scale()
    {
        GameObject obj = GameObject.FindWithTag(HexGrid.TAG);
        HexGrid manager = obj.GetComponent<HexGrid>();
        return manager.Scale;
    }
}
