# 🤖 Model Merger Authentication Integration

This guide covers the integration of authentication into Metatron2's Model Merger system.

## 🎯 What's Integrated

### **✅ Backend Authentication**
- **Protected API Routes** - All Model Merger endpoints require authentication
- **User-Specific Experiments** - Each user sees only their own merge experiments
- **Usage Tracking** - Track model merging resource consumption
- **Experiment Isolation** - Complete separation between users

### **✅ Frontend Integration**
- **Authentication Check** - Users must sign in to access Model Merger
- **User Context** - User information passed to Model Merger interface
- **Health Monitoring** - Check service availability with user tracking
- **Graceful Fallback** - Clear messaging when authentication is missing

### **✅ Database Integration**
- **User-Specific Storage** - Experiments stored with user context
- **Resource Tracking** - Monitor compute usage per user
- **Experiment History** - Complete audit trail of merge operations

## 🔧 How It Works

### **Authentication Flow**
1. **User clicks Model Merger** in main interface
2. **Authentication check** - Verify user is signed in
3. **Health check** - Ensure Model Merger service is available
4. **Load with context** - Pass user info to Model Merger interface
5. **Track usage** - Log access and resource consumption

### **URL Parameters Passed to Model Merger**
```
http://localhost:8000/model-merger/model-merger.html?user_id=123&user_email=user@example.com
```

### **API Endpoints (All Authenticated)**
- `GET /api/model-merger/health` - Check service status
- `GET /api/model-merger/models` - Get available models
- `GET /api/model-merger/methods` - Get merge methods
- `POST /api/model-merger/experiments` - Create merge experiment
- `GET /api/model-merger/experiments` - List user's experiments
- `GET /api/model-merger/experiments/{id}` - Get experiment details
- `GET /api/model-merger/experiments/{id}/download` - Download merged model

## 🧪 Testing the Integration

### **Test 1: Unauthenticated Access**
1. **Sign out** of Metatron2
2. **Click Model Merger** button
3. **Verify**: Login modal appears
4. **Expected**: Cannot access without authentication

### **Test 2: Authenticated Access**
1. **Sign in** to Metatron2
2. **Click Model Merger** button
3. **Verify**: 
   - Health check succeeds
   - Model Merger overlay opens
   - User context is displayed
   - Available models load
4. **Check browser console** for authentication logs

### **Test 3: Create Experiment**
1. **Open Model Merger** (authenticated)
2. **Select 2+ models** for merging
3. **Choose merge method** (e.g., "linear")
4. **Start merge experiment**
5. **Verify**:
   - Experiment appears in user's list
   - Usage is tracked in database
   - Progress updates work correctly

### **Test 4: User Isolation**
1. **Create experiments** as User A
2. **Sign out and sign in** as User B
3. **Open Model Merger**
4. **Verify**: User B cannot see User A's experiments

### **Test 5: Usage Tracking**
1. **Create multiple experiments**
2. **Check database** for usage events:
```sql
SELECT * FROM usage_events 
WHERE user_id = 'your-user-id' 
AND event_type LIKE '%model_merge%';
```
3. **Verify**: Events are logged with correct metadata

## 🔍 Troubleshooting

### **Model Merger Won't Load**
```javascript
// Check in browser console
console.log('Auth status:', window.authManager?.isAuthenticated());
```

**Solutions**:
- Ensure user is signed in
- Check if Model Merger service is running on port 5007
- Verify health check endpoint responds

### **No User Context in Model Merger**
```javascript
// Check URL parameters in Model Merger
console.log('URL params:', window.location.search);
```

**Solutions**:
- Verify user info is available in main app
- Check URL parameter formatting
- Ensure iframe src is set correctly

### **Experiments Not Saving**
```bash
# Check if Model Merger API is running
curl http://localhost:5007/api/model-merger/health
```

**Solutions**:
- Start Model Merger API: `cd backend/model-merger && python start.py`
- Check database connection
- Verify authentication tokens are valid

### **Service Unavailable**
```bash
# Check Model Merger service
curl http://localhost:8000/api/model-merger/health
```

**Solutions**:
- Ensure backend API is running
- Check if Model Merger service is properly integrated
- Verify port configurations

## 📊 Usage Analytics

The system tracks:
- **Model Merger Access** - When users open the tool
- **Experiment Creation** - New merge experiments with metadata
- **Resource Usage** - Compute time and model downloads
- **Success/Failure Rates** - Experiment completion statistics

## 🔄 Model Merger Service Architecture

### **Current Setup**
```
Main Metatron2 Backend (Port 8000)
├── /api/model-merger/* routes (authenticated)
└── Proxies to Model Merger Service (Port 5007)

Model Merger Service (Port 5007)
├── Standalone Flask API
├── Model downloading and merging
└── Experiment management
```

### **Authentication Flow**
```
User Request → Metatron2 API → Authentication Check → Model Merger Service
```

## 🚀 Production Considerations

### **Security**
- **API Key Management** - Secure Hugging Face API keys
- **Resource Limits** - Prevent abuse of compute resources
- **File Storage** - Secure storage for merged models
- **Rate Limiting** - Limit concurrent merge operations

### **Performance**
- **Queue Management** - Handle multiple merge requests
- **Resource Monitoring** - Track GPU/CPU usage
- **Storage Optimization** - Clean up old experiments
- **Caching** - Cache frequently used models

### **Scalability**
- **Distributed Processing** - Scale merge operations
- **Load Balancing** - Multiple Model Merger instances
- **Database Optimization** - Efficient experiment storage
- **CDN Integration** - Fast model downloads

## 🔧 Advanced Features

### **Experiment Sharing**
```javascript
// Future: Share experiments between users
await shareExperiment(experimentId, targetUserId);
```

### **Model Library Management**
```javascript
// Future: User-specific model libraries
await addCustomModel(modelPath, userMetadata);
```

### **Batch Operations**
```javascript
// Future: Batch merge multiple model combinations
await createBatchExperiment(modelCombinations);
```

## 📝 Development Notes

### **Adding New Merge Methods**
1. **Update backend** merge method definitions
2. **Add frontend** UI for method-specific parameters
3. **Test** with authentication and user isolation
4. **Document** new method capabilities

### **Custom Model Support**
1. **Implement** user model upload
2. **Add** model validation and storage
3. **Integrate** with existing merge workflows
4. **Ensure** proper user isolation

---

**🎉 Model Merger is now fully integrated with Metatron2 authentication!**

Users can securely create and manage model merge experiments with complete user isolation and comprehensive usage tracking.
